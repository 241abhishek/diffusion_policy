#!/usr/bin/env python
import rospy
import csv
from std_msgs.msg import String

def shutdown_hook():
    rospy.loginfo("Shutting down csv_publisher_node")

def csv_publisher():
    rospy.init_node('csv_publisher_node', anonymous=True)
    pub = rospy.Publisher('csv_topic', String, queue_size=10)
    rate = rospy.Rate(333.0/5)
    
    rospy.on_shutdown(shutdown_hook)

    try:
        with open('/home/cerebro/diff/data/synced_data/X2_SRA_A_07-05-2024_10-39-10-mod-sync.csv', 'r') as csvfile1, \
             open('/home/cerebro/diff/data/synced_data/X2_SRA_B_07-05-2024_10-41-46-mod-sync.csv', 'r') as csvfile2:
            csv_reader1 = csv.reader(csvfile1)
            csv_reader2 = csv.reader(csvfile2)
            line_counter = 0
            
            while not rospy.is_shutdown():
                try:
                    row1 = next(csv_reader1)
                    row2 = next(csv_reader2)
                except StopIteration:
                    rospy.loginfo("Reached end of CSV files")
                    break

                line_counter += 1
                if line_counter % 5 == 0:
                    row1_data = ','.join(row1[1:])
                    row2_data = ','.join(row2[1:])
                    combined_data = row1_data + ',' + row2_data
                    msg = String(data=combined_data)
                    pub.publish(msg)
                    # rospy.loginfo(f'Published: {msg.data}')
                    rate.sleep()
    finally:
        rospy.loginfo("CSV publisher has finished")

if __name__ == '__main__':
    try:
        csv_publisher()
    except rospy.ROSInterruptException:
        pass