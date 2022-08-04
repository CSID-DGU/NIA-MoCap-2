import torch
from lie.lie_util import *
from torch import nn


class LieSkeleton(object):
    def __init__(self, raw_translation, kinematic_tree, tensor):
        super(LieSkeleton, self).__init__()
        self.tensor = tensor
        # print(self.tensor)
        self._raw_translation = self.tensor(raw_translation.shape).copy_(raw_translation).detach()
        self._kinematic_tree = kinematic_tree
        self._translation = None
        self._parents = [0] * len(self._raw_translation)
        self._parents[0] = -1
        for chain in self._kinematic_tree:
            for j in range(1, len(chain)):
                self._parents[chain[j]] = chain[j-1]
                # 앞의 chain(척추, 팔, 다리)를 부모로 삼음

    def njoints(self):
        return len(self._raw_translation)
        # joint 수

    def raw_translation(self):
        return self._raw_translation
        # joint의 초기 위치

    def kinematic_tree(self):
        return self._kinematic_tree

    def parents(self):
        return self._translation

    def get_translation_joints(self, joints):
        # joint 위치 반환
        # joints/offsets (batch_size, joints_num, 3)
        # print(self._raw_translation.shape)
        _translation = self._raw_translation.clone().detach()
        # detach : 이후 연산의 추적 방지
	
        _translation = _translation.expand(joints.shape[0], -1, -1).clone()
        #print(_translation.shape)
        #print(self._raw_translation.shape)
        for i in range(1, self._raw_translation.shape[0]):
            _translation[:, i, :] = torch.norm(joints[:, i, :] - joints[:, self._parents[i], :], p=2, dim=1)[:, None] * \
                                     _translation[:, i, :]
        self._translation = _translation
        return _translation
        # _translation은 raw_translation으로부터 만든 행렬 -> joint 좌표의 이동을 뜻함

    def get_translation_bone(self, bonelengths):
        # bonelength (batch_size, joints_num - 1)
        # offsets (batch_size, joints_num, 3)
        # offset : 요소와 요소 사이의 변위차를 뜻함
	
        self._translation = self._raw_translation.clone().detach().expand(bonelengths.size(0), -1, -1).clone().to(bonelengths.device)
        self._translation[:, 1:, :] = bonelengths * self._translation[:, 1:, :]
        #  뼈의 길이가 _translation에 영향을 줌

    def inverse_kinemetics(self, joints):
        # 전체적인 형상을 통해 중간 joint들의 각도를 알아냄

        # joints (batch_size, joints_num, 3)
        # lie_params (batch_size, joints_num, 3)
        lie_params = self.tensor(joints.shape).fill_(0)
        # 각도 초기 지정
	
        # root_matR (batch_size, 3, 3)
        root_matR = torch.eye(3, dtype=joints.dtype).expand((joints.shape[0], -1, -1)).clone().detach().to(joints.device)
        # 초기 rotation matrix 지정
	
        for chain in self._kinematic_tree:
            R = root_matR
            for j in range(len(chain) - 1):
                # (batch, 3)
                u = self._raw_translation[chain[j + 1]].expand(joints.shape[0], -1).clone().detach().to(joints.device)
                # device : 해당 tensor가 cpu, gpu중 어디에 있는지 확인할 때 씀
		
		# u 벡터 정의
                # (batch, 3)
                v = joints[:, chain[j+1], :] - joints[:, chain[j], :]
                #  v 벡터 정의
                # (batch, 3)
                v = v / torch.norm(v, p=2, dim=1)[:, None]
                # (batch, 3, 3)
                R_local = torch.matmul(R.transpose(1, 2), lie_exp_map(lie_u_v(u, v)))
                # 반복문에 의해 rotation matrix가 계속 곱해짐
                # print("R_local shape:" + str(R_local.shape))
                # print(R_local)
                lie_params[:, chain[j + 1], :] = matR_log_map(R_local)
                R = torch.matmul(R, R_local)
        return lie_params
        # 반환값은 3x3 형태

    def forward_kinematics(self, lie_params, joints, root_translation, do_root_R = False, scale_inds=None):
        # 중간 joint들의 각도로 전체적인 형상 결정 - 뼈의 길이 정보 필요 없다
	
        # lie_params (batch_size, joints_num, 3) lie_params[:, 0, :] is not used
        # joints (batch_size, joints_num, 3)
        # root_translation (batch_size, 3)
        # translation_mat (batch_size, joints_num, 3)
        translation_mat = self.get_translation_joints(joints)
        if scale_inds is not None:
            # scale_inds : 추후 확인
	
            translation_mat[:, scale_inds, :] *= 1.25
        joints = self.tensor(lie_params.size()).fill_(0)
        joints[:, 0] = root_translation
        for chain in self._kinematic_tree:
            # if do_root_R is true, root coordinate system has rotation angulers
            # Plus, for chain not containing root(e.g arms), we use root rotation as the rotation
            # of joints near neck(i.e. beginning of this chain).
            if do_root_R:
                matR = lie_exp_map(lie_params[:, 0, :])
                # do_root_R이 true면(joint의 방향에 변화가 있다면) lie_params로 rotation matrix 뽑아냄

            # Or, root rotation matrix is identity matrix, which means no rotation at global coordinate system
            else:
                matR = torch.eye(3, dtype=joints.dtype).expand((joints.shape[0], -1, -1)).clone().detach().to(joints.device)
            for i in range(1, len(chain)):
                matR = torch.matmul(matR, lie_exp_map(lie_params[:, chain[i], :]))
                # rotation matrix가 계속 곱해짐
		
                translation_vec = translation_mat[:, chain[i], :].unsqueeze_(-1)
                joints[:, chain[i], :] = torch.matmul(matR, translation_vec).squeeze_()\
                                         + joints[:, chain[i-1], :]
        return joints
        # 3D location of joint -> 논문의 최종식
        # a2m 논문 Disentangled Lie Algebra Representation의 (4)식
    
