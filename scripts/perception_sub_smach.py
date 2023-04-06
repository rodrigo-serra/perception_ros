#!/usr/bin/env python3

import rospy
import smach
from smach import State,StateMachine

from perception_states import *


class WelcomeGuestSmach(smach.StateMachine):
    def __init__(self, success, failure):
        StateMachine.__init__(self, outcomes=["success", "failure"])

        with self:
            smach.StateMachine.add('TIAGO_ASK_NAME', PrintMsg("What is your name?"), transitions={'success': 'TIAGO_LISTEN_NAME', 'failure': 'TIAGO_ASK_NAME'})

            smach.StateMachine.add('TIAGO_LISTEN_NAME', FindName(), transitions={'success': 'TIAGO_SAY_NAME', 'failure': 'TIAGO_LISTEN_NAME'}, remapping={'out_data':'sentence'})

            smach.StateMachine.add('TIAGO_SAY_NAME', PrintMsg("", useInput = True), transitions={'success': 'TIAGO_ASK_DRINK', 'failure': 'TIAGO_SAY_NAME'}, remapping={'in_data': 'sentence'})

            smach.StateMachine.add('TIAGO_ASK_DRINK', PrintMsg("What is your favourite drink?"), transitions={'success': 'TIAGO_LISTEN_DRINK', 'failure': 'TIAGO_ASK_DRINK'})

            smach.StateMachine.add('TIAGO_LISTEN_DRINK', FindFavouriteDrink(), transitions={'success': 'TIAGO_SAY_DRINK', 'failure': 'TIAGO_LISTEN_DRINK'}, remapping={'out_data':'sentence'})

            smach.StateMachine.add('TIAGO_SAY_DRINK', PrintMsg("", useInput = True), transitions={'success': 'TIAGO_SAY_PHOTO', 'failure': 'TIAGO_SAY_DRINK'}, remapping={'in_data': 'sentence'})

            smach.StateMachine.add('TIAGO_SAY_PHOTO', PrintMsg("I am about to take you a photo so I can remenber you"), transitions={'success': 'TIAGO_CHECK_DETECTION', 'failure': 'TIAGO_SAY_PHOTO'})

            smach.StateMachine.add('TIAGO_CHECK_DETECTION', SeeingPeople(), transitions={'success': 'TAKE_PHOTO_REID', 'failure': 'TIAGO_CHECK_DETECTION_FAILURE'})

            smach.StateMachine.add('TIAGO_CHECK_DETECTION_FAILURE', PrintMsg("I can not see your face properly. Please get in front of me"), transitions={'success': 'WAIT_FOR_PERSON_ADJUSTMENT', 'failure': 'TIAGO_CHECK_DETECTION_FAILURE'})

            smach.StateMachine.add('WAIT_FOR_PERSON_ADJUSTMENT', Sleep(0.5),transitions={'success': 'TIAGO_CHECK_DETECTION_ATTEMPT'})

            smach.StateMachine.add('TIAGO_CHECK_DETECTION_ATTEMPT', PrintMsg("Let's try again"), transitions={'success': 'TIAGO_CHECK_DETECTION', 'failure': 'TIAGO_CHECK_DETECTION_ATTEMPT'})

            smach.StateMachine.add('TAKE_PHOTO_REID', send_event([('/perception/reid/event_in', 'e_take_photo')]), transitions={'success': 'SAVING_REID_INFO'})

            smach.StateMachine.add('SAVING_REID_INFO', SavePersonInfo(), transitions={'success': 'success', 'failure': 'TIAGO_CHECK_DETECTION_FAILURE'})