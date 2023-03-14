#!/usr/bin/env python3

import rospy
import smach
import smach_ros
import threading
import random

# perception.py script
from perception import *

perception_object = Perception() 


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



if __name__ == '__main__':
    rospy.init_node('my_smach_state_machine')
    # Create a SMACH state machine
    sm = smach.StateMachine(outcomes=['success', 'failure'])
    # Open the container
    with sm:
        #Add states to the container
        smach.StateMachine.add('TIAGO_INFO_INTRO', PrintMsg("Please point to one of the bags"), transitions={'success': 'POINTING_DIRECTION_PERSON', 'failure': 'TIAGO_INFO_INTRO'})
        
        smach.StateMachine.add('POINTING_DIRECTION_PERSON', DetectPointingDirection(), transitions={'left': 'TIAGO_SAY_LEFT', 'right': 'TIAGO_SAY_RIGHT', 'failure': 'TIAGO_SAY_FAILURE'})

        smach.StateMachine.add('TIAGO_SAY_LEFT', PrintMsg("You are pointing to the bag at the left"), transitions={'success': 'success', 'failure': 'TIAGO_SAY_FAILURE'})
        smach.StateMachine.add('TIAGO_SAY_RIGHT', PrintMsg("You are pointing to the bag at the right"), transitions={'success': 'success', 'failure': 'TIAGO_SAY_FAILURE'})
        smach.StateMachine.add('TIAGO_SAY_FAILURE', PrintMsg("Could not get pointing direction. Let's do it again"), transitions={'success': 'TIAGO_INFO_INTRO', 'failure': 'TIAGO_SAY_FAILURE'})

    # Execute SMACH plan
    outcome = sm.execute()
