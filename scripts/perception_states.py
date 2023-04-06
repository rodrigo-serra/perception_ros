#!/usr/bin/env python3

import rospy
import smach
import std_msgs

# perception.py script
from perception import *
perception_object = Perception() 

from perception_states_db import SemanticMapping, Person
# Semantic Mapping Global Variable Initialization
semanticMap = SemanticMapping()


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

    def __init__(self, msg, useInput = False):
        smach.State.__init__(self, outcomes=['success', 'failure'], input_keys=['in_data'])
        self.msg = msg
        self.useInput = useInput

    def execute(self,userdata):
        try:
            if self.useInput:
                rospy.loginfo(userdata.in_data)
            else:
                rospy.loginfo(self.msg)
            return 'success'
        except:
            return 'failure'


class FindName(smach.State):
    '''
    description: it reads a msg
    input: 
    outcomes: 'success' or 'failure'
    '''

    def __init__(self):
        smach.State.__init__(self, outcomes=['success', 'failure'], output_keys=['out_data'])

    def execute(self,userdata):
        try:
            firstName = input('Enter your name: ')
        except:
            userdata.out_data = ""
            return 'failure'

        userdata.out_data = "Nice to meet you " + firstName
        
        # Add Person to the Semantic Mapping for the first time
        p = Person()
        p.name = firstName
        p.party_id = "guest#" + str(len(semanticMap.persons) + 1)
        semanticMap.persons.append(p)
        
        return 'success'


class FindFavouriteDrink(smach.State):
    '''
    description: it reads a msg
    input: 
    outcomes: 'success' or 'failure'
    '''

    def __init__(self):
        smach.State.__init__(self, outcomes=['success', 'failure'], output_keys=['out_data'])

    def execute(self,userdata):
        try:
            favDrink = input('Enter your favourite drink: ')
        except:
            userdata.out_data = ""
            return 'failure'

        userdata.out_data = "Your favourite drink is " + favDrink
        
        # Add favDrink 
        semanticMap.persons[len(semanticMap.persons) - 1].favorite_drink = favDrink
        
        return 'success'


class SeeingPeople(smach.State):
    '''
    description: See if reid is detecting the person
    input: 
    outcomes: 'success' or 'failure'
    '''

    def __init__(self):
        smach.State.__init__(self, outcomes=['success', 'failure'])

    def execute(self,userdata):
        try:
            detection = perception_object.getPeopleDetection()
        except:
            return 'failure'

        if detection == None:
            return 'failure'
        
        # rospy.logwarn(str(detection))
        return 'success'


class SavePersonInfo(smach.State):
    '''
    description: Add Person info to the global database
    input: 
    outcomes: 'success' or 'failure'
    '''

    def __init__(self):
        smach.State.__init__(self, outcomes=['success', 'failure'])

    def execute(self,userdata):
        try:
            detection = perception_object.getClosestPersonToCamera()
        except:
            return 'failure'

        if detection == None:
            return 'failure'
        
        if semanticMap.persons[len(semanticMap.persons) - 1].reid_id == "Unknown":
            return 'failure'
        
        semanticMap.persons[len(semanticMap.persons) - 1].reid_id = detection.id
        semanticMap.persons[len(semanticMap.persons) - 1].age_range = detection.ageRange
        semanticMap.persons[len(semanticMap.persons) - 1].gender = detection.gender

        rospy.logwarn(str(semanticMap.persons[len(semanticMap.persons) - 1].name))
        rospy.logwarn(str(semanticMap.persons[len(semanticMap.persons) - 1].favorite_drink))
        rospy.logwarn(str(semanticMap.persons[len(semanticMap.persons) - 1].reid_id))
        rospy.logwarn(str(semanticMap.persons[len(semanticMap.persons) - 1].party_id))
        rospy.logwarn(str(semanticMap.persons[len(semanticMap.persons) - 1].age_range))
        rospy.logwarn(str(semanticMap.persons[len(semanticMap.persons) - 1].gender))

        return 'success'




#################################################################################################
class send_event(smach.State):
    '''
    This state will take a list of event as input. Which are pair of name and value to publish.
    Output of this node is to publish the value in the provided topic name.
    '''

    def __init__(self, event_list):
        smach.State.__init__(self, outcomes=['success'])
        self.event_publisher_list = []
        self.expected_return_values_ = []
        self.event_names_ = []
        self.possible_event_values = ['e_start', 'e_stop', 'e_trigger', 'e_forget', 'e_take_photo']
        for event in event_list:
            if len(event) != 2:
                rospy.logerr('The event list is malformed!!')
                exit()
            elif event[1].lower() not in self.possible_event_values:
                rospy.logerr('Improper event value!!')
                exit()

            event_name = event[0]
            self.event_names_.append(event_name)
            self.expected_return_values_.append(event[1].lower())
            self.event_publisher_list.append(rospy.Publisher(event_name, std_msgs.msg.String, queue_size=1))

        # give some time for the publishers to register in the network
        rospy.sleep(0.1)

    def execute(self, userdata):
        for index in range(len(self.event_publisher_list)):
            self.event_publisher_list[index].publish(self.expected_return_values_[index])
            rospy.logdebug('Published the event_name: %s event_value: %s', self.event_names_[index],
                           self.expected_return_values_[index])
        return 'success'


class Sleep(smach.State):
    '''
    This state just sleeps for some time
    '''
    def __init__(self, time_to_sleep):
        smach.State.__init__(self, outcomes=['success'])
        self.time_to_sleep = time_to_sleep

    def execute(self, userdata):
        rospy.loginfo('sleeping for ' + str(self.time_to_sleep) + ' seconds' )
        rospy.sleep(self.time_to_sleep)
        return 'success'

