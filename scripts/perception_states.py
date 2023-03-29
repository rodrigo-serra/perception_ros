#!/usr/bin/env python3

import rospy
import smach
import smach_ros
import threading
import random

# perception.py script
from perception import *

perception_object = Perception() 


class CurrentPeopleDetection(smach.State):
    '''
    description: it returns the direction someone is pointing at
    input: 
    outcomes: 'left', 'right' or 'failure'
    '''

    def __init__(self):
        smach.State.__init__(self, outcomes=['success', 'failure'], input_keys=['in_data'])
        self.return_phrase = None
        self.ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        self.ageList2 = ['0 and 2', '4 and 6', '8 and 12', '15 and 20', '25 and 32', '38 and 43', '48 and 53', '60 and 100']
        self.maleAgeCounter = [0, 0, 0, 0, 0, 0, 0, 0]
        self.femaleAgeCounter = [0, 0, 0, 0, 0, 0, 0, 0]
        self.genderList = ['Male', 'Female']

    def execute(self,userdata):
        detectionList = perception_object.getPeopleDetection()
        if detectionList == 'None':
            self.return_phrase = "My eyes are not seeing anyone"
            return 'failute'
        else: 
            # self.return_phrase = ""
            # for idx, d in enumerate(detectionList):
            #     self.return_phrase += "I see a " + d.gender + " with an estimated age between "
            #     ageSplit = (d.ageRange).split("-")
            #     ageMin = ageSplit[0].split("(")[1]
            #     ageMax = ageSplit[1].split(")")[0]
            #     self.return_phrase += ageMin + " and " + ageMax
            #     if len(detectionList) != 1:
            #         if idx == len(detectionList) - 2:
            #             self.return_phrase += " and "
            #         elif idx == len(detectionList) - 1:
            #             self.return_phrase += "."
            #         else:
            #             self.return_phrase += ", "
            #     else:
            #         self.return_phrase += "."
            
            # rospy.logwarn(self.return_phrase)
            # return 'success'
            
            for idx, d in enumerate(detectionList):
                ageIdx = self.ageList.index(d.ageRange)
                if d.gender == self.genderList[0]:
                    self.maleAgeCounter[ageIdx] += 1
                else:
                    self.femaleAgeCounter[ageIdx] += 1

            self.return_phrase = ""
            writeCtr = False
            for i in range(len(self.maleAgeCounter)):
                if self.maleAgeCounter[i] != 0 and self.femaleAgeCounter[i] != 0:
                    if self.maleAgeCounter[i] > 0:
                        self.return_phrase += "I see " + str(self.maleAgeCounter[i]) + " men and "
                    else:
                        self.return_phrase += "I see one man and "

                    if self.femaleAgeCounter[i] > 0:
                        self.return_phrase += str(self.femaleAgeCounter[i]) + " women with and estimated age between " + self.ageList2[i]
                    else:
                        self.return_phrase += "one woman with an estimated age between " + self.ageList2[i]

                    writeCtr = True
                    
                else:
                    if self.maleAgeCounter[i] != 0:
                        if self.maleAgeCounter[i] > 0:
                            self.return_phrase += "I see " + str(self.maleAgeCounter[i]) + " men with an estimated age between " + self.ageList2[i]
                        else:
                            self.return_phrase += "I see one woman with an estimated age between " + self.ageList2[i]

                        writeCtr = True
                    
                    if self.femaleAgeCounter[i] != 0:
                        if self.femaleAgeCounter[i] > 0:
                            self.return_phrase += "I see " + str(self.femaleAgeCounter[i]) + " women with an estimated age between " + self.ageList2[i]
                        else:
                            self.return_phrase += "I see one woman with an estimated age between " + self.ageList2[i]
                        
                        writeCtr = True
                
                if self.return_phrase != "" and writeCtr == True:
                    ctr = 0
                    for j in range(i + 1, len(self.maleAgeCounter)):
                        if self.maleAgeCounter[j] != 0 or self.femaleAgeCounter[j] != 0:
                            ctr += 1
                    
                    if ctr == 0:
                        self.return_phrase += "."

                    if ctr == 1:
                        self.return_phrase += ", and "

                    if ctr > 1:
                        self.return_phrase += ", "
                
                writeCtr = False

            rospy.logwarn(self.return_phrase)
            return 'success'


class DetectPointingDirection(smach.State):
    '''
    description: it returns the direction someone is pointing at
    input: 
    outcomes: 'left', 'right' or 'failure'
    '''

    def __init__(self):
        smach.State.__init__(self, outcomes=['left', 'right','failure'], input_keys=['in_data'])

    def execute(self,userdata):
        pointingDirection = perception_object.getPointingDirection()

        if pointingDirection == 'left':
            return 'left'
        elif pointingDirection == 'right':
            return 'right'
        else: 
            return 'failure'


class PrintMsg(smach.State):
    '''
    description: it prints a msg
    input: 
    outcomes: 'success' or 'failure'
    '''

    def __init__(self, msg):
        smach.State.__init__(self, outcomes=['success', 'failure'], input_keys=['in_data'])
        self.msg = msg

    def execute(self,userdata):
        try:
            rospy.loginfo(self.msg)
            return "success"
        except:
            return "failure"


# POINTING DIRECTION SMACH
# if __name__ == '__main__':
#     rospy.init_node('my_smach_state_machine')
#     # Create a SMACH state machine
#     sm = smach.StateMachine(outcomes=['success', 'failure'])
#     # Open the container
#     with sm:
#         #Add states to the container
#         smach.StateMachine.add('TIAGO_INFO_INTRO', PrintMsg("Please point to one of the bags"), transitions={'success': 'POINTING_DIRECTION_PERSON', 'failure': 'TIAGO_INFO_INTRO'})
        
#         smach.StateMachine.add('POINTING_DIRECTION_PERSON', DetectPointingDirection(), transitions={'left': 'TIAGO_SAY_LEFT', 'right': 'TIAGO_SAY_RIGHT', 'failure': 'TIAGO_SAY_FAILURE'})

#         smach.StateMachine.add('TIAGO_SAY_LEFT', PrintMsg("You are pointing to the bag at the left"), transitions={'success': 'success', 'failure': 'TIAGO_SAY_FAILURE'})
#         smach.StateMachine.add('TIAGO_SAY_RIGHT', PrintMsg("You are pointing to the bag at the right"), transitions={'success': 'success', 'failure': 'TIAGO_SAY_FAILURE'})
#         smach.StateMachine.add('TIAGO_SAY_FAILURE', PrintMsg("Could not get pointing direction. Let's do it again"), transitions={'success': 'TIAGO_INFO_INTRO', 'failure': 'TIAGO_SAY_FAILURE'})

#     # Execute SMACH plan
#     outcome = sm.execute()


# PEOPLE DETECTION SMACH
if __name__ == '__main__':
    rospy.init_node('my_smach_state_machine')
    # Create a SMACH state machine
    sm = smach.StateMachine(outcomes=['success', 'failure'])
    # Open the container
    with sm:
        #Add states to the container
        smach.StateMachine.add('TIAGO_INFO_INTRO', PrintMsg("Hello there"), transitions={'success': 'DETECT_PERSONS', 'failure': 'TIAGO_INFO_INTRO'})
        
        smach.StateMachine.add('DETECT_PERSONS', CurrentPeopleDetection(), transitions={'success': 'TIAGO_SAY_FINAL_MSG', 'failure': 'DETECT_PERSONS'})

        smach.StateMachine.add('TIAGO_SAY_FINAL_MSG', PrintMsg("I successfuly identified people"), transitions={'success': 'success', 'failure': 'TIAGO_SAY_FINAL_MSG'})

    # Execute SMACH plan
    outcome = sm.execute()
