#!/usr/bin/env python3
import rclpy
import time
import sys
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters

def main():
    rclpy.init()
    node = rclpy.create_node('gz_gain_setter')

    target_node = '/gz_ros2_control'
    param_name = 'position_proportional_gain'
    param_value = 12.0

    client = node.create_client(SetParameters, f'{target_node}/set_parameters')

    # Wait for the target node to be available
    node.get_logger().info(f'Waiting for {target_node} node...')
    timeout = 30.0
    start = time.time()
    while not client.service_is_ready() and (time.time() - start) < timeout:
        time.sleep(0.5)

    if not client.service_is_ready():
        node.get_logger().error(f'{target_node} node not available after {timeout}s')
        return 1

    # Set the parameter
    request = SetParameters.Request()
    param = Parameter()
    param.name = param_name
    param.value.type = ParameterType.PARAMETER_DOUBLE
    param.value.double_value = param_value
    request.parameters = [param]

    node.get_logger().info(f'Setting {target_node}/{param_name} = {param_value}')
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future, timeout_sec=5.0)

    if future.done():
        response = future.result()
        if response.results[0].successful:
            node.get_logger().info(f'Successfully set {param_name} to {param_value}')
            return 0
        else:
            node.get_logger().error(f'Failed: {response.results[0].reason}')
            return 1
    else:
        node.get_logger().error('Service call timed out')
        return 1

if __name__ == '__main__':
    sys.exit(main())
