#!/usr/bin/env python3

import rospy
import smach

from perception_states import *
from perception_sub_smach import WelcomeGuestSmach


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
# if __name__ == '__main__':
#     rospy.init_node('my_smach_state_machine')
#     # Create a SMACH state machine
#     sm = smach.StateMachine(outcomes=['success', 'failure'])
#     # Open the container
#     with sm:
#         #Add states to the container
#         smach.StateMachine.add('TIAGO_INFO_INTRO', PrintMsg("Hello there"), transitions={'success': 'DETECT_PERSONS', 'failure': 'TIAGO_INFO_INTRO'})
        
#         smach.StateMachine.add('DETECT_PERSONS', CurrentPeopleDetection(), transitions={'success': 'TIAGO_SAY_FINAL_MSG', 'failure': 'DETECT_PERSONS'})

#         smach.StateMachine.add('TIAGO_SAY_FINAL_MSG', PrintMsg("I successfuly identified people"), transitions={'success': 'success', 'failure': 'TIAGO_SAY_FINAL_MSG'})

#     # Execute SMACH plan
#     outcome = sm.execute()


# WELCOME GUEST SMACH
if __name__ == '__main__':
    rospy.init_node('my_smach_state_machine')
    # Create a SMACH state machine
    sm = smach.StateMachine(outcomes=['success', 'failure'])
    # Open the container
    with sm:
        welcoming_guest_sub_sm = WelcomeGuestSmach(success='success', failure='failure')

        #Add states to the container
        smach.StateMachine.add('START_REID', send_event([('/perception/reid/event_in', 'e_start')]), transitions={'success': 'WAIT_FOR_REID_INIT'})

        smach.StateMachine.add('WAIT_FOR_REID_INIT', Sleep(0.5),transitions={'success': 'WELCOME_GUEST_1_SUB_SMACH'})

        smach.StateMachine.add('WELCOME_GUEST_1_SUB_SMACH', welcoming_guest_sub_sm, transitions={'success': 'GET_OK_1', 'failure': 'failure'})
        
        smach.StateMachine.add('GET_OK_1', PrintMsg("OK"), transitions={'success': 'WELCOME_GUEST_2_SUB_SMACH', 'failure': 'GET_OK_1'})

        smach.StateMachine.add('WELCOME_GUEST_2_SUB_SMACH', welcoming_guest_sub_sm, transitions={'success': 'GET_OK_2', 'failure': 'failure'})

        smach.StateMachine.add('GET_OK_2', PrintMsg("OK"), transitions={'success': 'success', 'failure': 'GET_OK_2'})
        
        

    # Execute SMACH plan
    outcome = sm.execute()
