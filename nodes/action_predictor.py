#!/usr/bin/env python

import rospy
import csv
from std_msgs.msg import String

def csv_publisher():
    rospy.init_node('csv_publisher_node', anonymous=True)

    # Define the topic to publish to and the message type
    pub = rospy.Publisher('csv_topic', String, queue_size=10)

    # Define the rate at which to publish (in Hz)
    rate = rospy.Rate(333.0/5)

    # Open the CSV file
    with open(\
        '/home/cerebro/diff/data/synced_data/X2_SRA_A_07-05-2024_10-39-10-mod-sync.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile)

        # Loop through the CSV file
        for row in csv_reader:
            # Convert the row to a string message (modify if using a different message type)
            msg = String()
            msg.data = ','.join(row)
            
            # Publish the message
            pub.publish(msg)
            
            # Log the published message
            rospy.loginfo(f'Published: {msg.data}')

            # Sleep to maintain the defined rate
            rate.sleep()

if __name__ == '__main__':
    try:
        csv_publisher()
    except rospy.ROSInterruptException:
        pass