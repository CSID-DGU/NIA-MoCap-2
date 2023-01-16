import os
import json
import csv
import numpy as np
import pandas as pd

csv_dir = 'path of bvh files'
json_dir = 'path of json files'

for file in os.listdir(csv_dir):
  if 'csv' in file:
    csv_path = os.path.join(csv_dir, file)
    name, ext = os.path.splitext(csv_path)
    # you should adjust appropriately indexes of name
    json_path = os.path.join(json_dir + name[16:-9] + '.json')
    print(csv_path, json_path)
    with open(json_path, encoding="UTF-8") as f:
      print("processing " + json_path + "...")
      # you should adjust appropriately indexes
      if(csv_path[17:-13] != json_path[18:-5]):
          print(json_path)
          print(csv_path)
          break
      json_obj = json.load(f)
      csv_file = pd.read_csv(csv_path)
      numofaction = len(json_obj["annotation"]["actionAnnotationList"])
      coordinates = []

      for i in range(numofaction):
        filename = "" + 'P' + '{0:02d}'.format(json_obj["annotation"]["actionAnnotationList"][i]["actor_id"]) + 'G' + '{0:02d}'.format(json_obj["annotation"]["actionAnnotationList"][i]["scenario_id"]) +'R' + '{0:02d}'.format(json_obj["annotation"]["actionAnnotationList"][i]["appearance_id"]) + 'F' + '{0:04d}'.format(json_obj["annotation"]["actionAnnotationList"][i]["start_frame"]) + 'T' + '{0:04d}'.format(json_obj["annotation"]["actionAnnotationList"][i]["end_frame"]) + 'A' + '{0:03d}'.format(json_obj["annotation"]["actionAnnotationList"][i]["action_id"])
        s_frame = json_obj["annotation"]["actionAnnotationList"][i]["start_frame"]
        f_frame = json_obj["annotation"]["actionAnnotationList"][i]["end_frame"]
        x=[]
        for j in range(s_frame, f_frame - 4, 4):
          try:
            coordinates += [[[csv_file['Hips.X'][j]/100, csv_file['Hips.Y'][j]/100, csv_file['Hips.Z'][j]/100],
            [csv_file['LeftUpLeg.X'][j]/100, csv_file['LeftUpLeg.Y'][j]/100, csv_file['LeftUpLeg.Z'][j]/100],
            [csv_file['RightUpLeg.X'][j]/100, csv_file['RightUpLeg.Y'][j]/100, csv_file['RightUpLeg.Z'][j]/100],
            
            [csv_file['Spine.X'][j]/100, csv_file['Spine.Y'][j]/100, csv_file['Spine.Z'][j]/100],
            [csv_file['LeftLeg.X'][j]/100, csv_file['LeftLeg.Y'][j]/100, csv_file['LeftLeg.Z'][j]/100],
            [csv_file['RightLeg.X'][j]/100, csv_file['RightLeg.Y'][j]/100, csv_file['RightLeg.Z'][j]/100],
            
            [csv_file['Spine1.X'][j]/100, csv_file['Spine1.Y'][j]/100, csv_file['Spine1.Z'][j]/100],
            [csv_file['LeftFoot.X'][j]/100, csv_file['LeftFoot.Y'][j]/100, csv_file['LeftFoot.Z'][j]/100],
            [csv_file['RightFoot.X'][j]/100, csv_file['RightFoot.Y'][j]/100, csv_file['RightFoot.Z'][j]/100],
            
            [csv_file['Spine2.X'][j]/100, csv_file['Spine2.Y'][j]/100, csv_file['Spine2.Z'][j]/100],
            [csv_file['LeftToeBase.X'][j]/100, csv_file['LeftToeBase.Y'][j]/100, csv_file['LeftToeBase.Z'][j]/100],
            [csv_file['RightToeBase.X'][j]/100, csv_file['RightToeBase.Y'][j]/100, csv_file['RightToeBase.Z'][j]/100],
            
            [csv_file['Neck.X'][j]/100, csv_file['Neck.Y'][j]/100, csv_file['Neck.Z'][j]/100],
            [csv_file['LeftShoulder.X'][j]/100, csv_file['LeftShoulder.Y'][j]/100, csv_file['LeftShoulder.Z'][j]/100],
            [csv_file['RightShoulder.X'][j]/100, csv_file['RightShoulder.Y'][j]/100, csv_file['RightShoulder.Z'][j]/100],
            
            [csv_file['Head.X'][j]/100, csv_file['Head.Y'][j]/100, csv_file['Head.Z'][j]/100],
            [csv_file['LeftArm.X'][j]/100, csv_file['LeftArm.Y'][j]/100, csv_file['LeftArm.Z'][j]/100],
            [csv_file['RightArm.X'][j]/100, csv_file['RightArm.Y'][j]/100, csv_file['RightArm.Z'][j]/100],
            [csv_file['LeftForeArm.X'][j]/100, csv_file['LeftForeArm.Y'][j]/100, csv_file['LeftForeArm.Z'][j]/100],
            [csv_file['RightForeArm.X'][j]/100, csv_file['RightForeArm.Y'][j]/100, csv_file['RightForeArm.Z'][j]/100],
            [csv_file['LeftForeArmRoll.X'][j]/100, csv_file['LeftForeArmRoll.Y'][j]/100, csv_file['LeftForeArmRoll.Z'][j]/100],
            [csv_file['RightForeArmRoll.X'][j]/100, csv_file['RightForeArmRoll.Y'][j]/100, csv_file['RightForeArmRoll.Z'][j]/100],
            [csv_file['LeftHand.X'][j]/100, csv_file['LeftHand.Y'][j]/100, csv_file['LeftHand.Z'][j]/100],
            [csv_file['RightHand.X'][j]/100, csv_file['RightHand.Y'][j]/100, csv_file['RightHand.Z'][j]/100]]]
          except Exception as e:
            print('Error Exception ', e)

          
        x = np.array(coordinates)
        np.save("/home/irteam/data/{}".format(filename), x)
        coordinates = []
      print(json_path + ' is done.')
      
