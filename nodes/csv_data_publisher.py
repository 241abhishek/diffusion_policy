#!/usr/bin/env python
import rospy
import csv
from std_msgs.msg import Float32MultiArray
from diff_policy.msg import X2RobotState

def shutdown_hook():
    rospy.loginfo("Shutting down csv_publisher_node")

def csv_publisher():
    """
    Publishes data from two CSV files line by line, concatenated together to simulate real-time sensor data from two exoskeletons. 
    """
    rospy.init_node('csv_publisher_node', anonymous=True)
    mode = 'csv_sim' # 'csv_sim' or 'real' (csv_sim for simulation with true action data also available at decimated rate, real for real-time sim of exo)

    if mode == 'csv_sim':
        pub = rospy.Publisher('csv_topic', Float32MultiArray, queue_size=10)
        decimation_rate = 5
        rate = rospy.Rate(333.0/decimation_rate)
    elif mode == 'real':
        pub = rospy.Publisher('/X2_SRA_A/custom_robot_state', X2RobotState, queue_size=10)
        decimation_rate = 1
        rate = rospy.Rate(333.0/decimation_rate)
    else:
        rospy.logerr("Invalid mode specified: must be 'csv_sim' or 'real'")
        # exit if invalid mode specified
        return
    
    rospy.on_shutdown(shutdown_hook)

    start_row, end_row = 696306, 894640  

    try:
        with open('/home/cerebro/diff/data/synced_data/X2_SRA_A_07-05-2024_10-39-10-mod-sync.csv', 'r') as csvfile1, \
             open('/home/cerebro/diff/data/synced_data/X2_SRA_B_07-05-2024_10-41-46-mod-sync.csv', 'r') as csvfile2:
            csv_reader1 = csv.reader(csvfile1)
            csv_reader2 = csv.reader(csvfile2)
            
            # Skip rows before start_row
            for _ in range(start_row):
                next(csv_reader1, None)
                next(csv_reader2, None)
            
            line_counter = start_row
            
            while not rospy.is_shutdown():
                if end_row is not None and line_counter >= end_row:
                    rospy.loginfo(f"Reached specified end row: {end_row}")
                    break
                
                try:
                    row1 = next(csv_reader1)
                    row2 = next(csv_reader2)
                except StopIteration:
                    rospy.loginfo("Reached end of CSV files")
                    break

                if line_counter % decimation_rate == 0:
                    # Convert row data to float and concatenate
                    row1_data = [float(x) for x in row1[1:]]
                    row2_data = [float(x) for x in row2[1:]]
                    combined_data = row1_data + row2_data
                    
                    if mode == 'csv_sim':
                        # Create Float32MultiArray message
                        msg = Float32MultiArray(data=combined_data)

                    elif mode == 'real':
                        # create X2RobotState message
                        combined_data = combined_data[:4] # first 4 values are the joint angles for the patient
                        combined_data.append(0.0) # add a dummy value for the imu
                        msg = X2RobotState()
                        msg.header.stamp = rospy.Time.now()
                        msg.joint_state.position = combined_data

                    pub.publish(msg)
                    # rospy.loginfo(f'Published row {line_counter}: {msg.data}')
                    rate.sleep()
                line_counter += 1
    finally:
        rospy.loginfo("CSV publisher has finished")

if __name__ == '__main__':
    try:
        csv_publisher()
    except rospy.ROSInterruptException:
        pass