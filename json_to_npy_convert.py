import os
import json
import numpy as np

dataset_dir = 'C:\\niamocap\\action-to-motion\dataset'
for file in os.listdir(dataset_dir):
  if 'json' in file:
    print(file)
    json_path = os.path.join(dataset_dir, file)
    with open(json_path, encoding="UTF-8") as f:
      json_obj = json.load(f)

  numofaction = len(json_obj["annotations"])

  coordinates = []
  for i in range(numofaction):
    filename = "" + json_obj["videos"][0]["mocap_filename"][3:7] + str(json_obj["annotations"][i]["action_id"]) + 'R' + json_obj["videos"][0]["mocap_filename"][16] +'P' + str(json_obj["annotations"][i]["actor_id"]) + 'F' + str(json_obj["annotations"][i]["start_frame"]) + 'T' + str(json_obj["annotations"][i]["end_frame"])
    s_frame = json_obj["mocap_data"][json_obj["annotations"][i]["start_frame"]]["Frame"]
    f_frame = json_obj["mocap_data"][json_obj["annotations"][i]["end_frame"]]["Frame"]
    
    for j in range(s_frame, f_frame):
      frame = json_obj["mocap_data"][j]
      coordinates += [[[frame["Hips_X"], frame["Hips_Y"], frame["Hips_Z"]], 
      [frame["LeftUpLeg_X"], frame["LeftUpLeg_Y"], frame["LeftUpLeg_Z"]], 
      [frame["RightUpLeg_X"], frame["RightUpLeg_Y"], frame["RightUpLeg_Z"]], 
      [frame["Spine_X"], frame["Spine_Y"], frame["Spine_Z"]], 
      [frame["LeftLeg_X"], frame["LeftLeg_Y"], frame["LeftLeg_Z"]], 
      [frame["RightLeg_X"], frame["RightLeg_Y"], frame["RightLeg_Z"]], 
      [frame["Spine1_X"], frame["Spine1_Y"], frame["Spine1_Z"]], 
      [frame["LeftFoot_X"], frame["LeftFoot_Y"], frame["LeftFoot_Z"]], 
      [frame["RightFoot_X"], frame["RightFoot_Y"], frame["RightFoot_Z"]], 
      [frame["Spine2_X"], frame["Spine2_Y"], frame["Spine2_Z"]], 
      [frame["LeftToeBase_X"], frame["LeftToeBase_Y"], frame["LeftToeBase_Z"]], 
      [frame["RightToeBase_X"], frame["RightToeBase_Y"], frame["RightToeBase_Z"]], 
      [frame["Neck_X"], frame["Neck_Y"], frame["Neck_Z"]], 
      [frame["LeftShoulder_X"], frame["LeftShoulder_Y"], frame["LeftShoulder_Z"]], 
      [frame["RightShoulder_X"], frame["RightShoulder_Y"], frame["RightShoulder_Z"]], 
      [frame["Head_X"], frame["Head_Y"], frame["Head_Z"]], 
      [frame["LeftArm_X"], frame["LeftArm_Y"], frame["LeftArm_Z"]], 
      [frame["RightArm_X"], frame["RightArm_Y"], frame["RightArm_Z"]], 
      [frame["LeftForeArm_X"], frame["LeftForeArm_Y"], frame["LeftForeArm_Z"]], 
      [frame["RightForeArm_X"], frame["RightForeArm_Y"], frame["RightForeArm_Z"]], 
      [frame["LeftForeArmRoll_X"], frame["LeftForeArmRoll_Y"], frame["LeftForeArmRoll_Z"]], 
      [frame["RightForeArmRoll_X"], frame["RightForeArmRoll_Y"], frame["RightForeArmRoll_Z"]], 
      [frame["LeftHand_X"], frame["LeftHand_Y"], frame["LeftHand_Z"]], 
      [frame["RightHand_X"], frame["RightHand_Y"], frame["RightHand_Z"]]]]

  x = np.array(coordinates)
  np.save("./{}".format(filename), x)
    
