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
    outcomes: 'success' or 'failure'
    '''

    def __init__(self):
        smach.State.__init__(self, outcomes=['success', 'failure'], input_keys=['in_data'])
        print("DetectPointingDirection init")

    def execute(self,userdata):
        pointingDirection = perception_object.getPointingDirection()

        if pointingDirection:
            rospy.loginfo("Pointing to the " + pointingDirection)
            return 'success'
        else: 
            return 'failure'


class RequestPointToBag(smach.State):
    '''
    description: it requests someone to point to a bag
    input: 
    outcomes: 'success' or 'failure'
    '''

    def __init__(self):
        smach.State.__init__(self, outcomes=['success', 'failure'], input_keys=['in_data'])
        self.msg = "Please point to one of the bags"
        print("RequestPointToBag init")

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
        smach.StateMachine.add('RequestPointToBag', RequestPointToBag(), transitions={'success': 'DetectPointingDirection', 'failure': 'RequestPointToBag'})
        smach.StateMachine.add('DetectPointingDirection', DetectPointingDirection(), transitions={'success': 'success', 'failure': 'RequestPointToBag'})

    # Execute SMACH plan
    outcome = sm.execute()
