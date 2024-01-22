from Rosmaster_Lib import Rosmaster 
from camera_Lib import Rosmaster_Camera
import serial

# 返回单片机版本号, tcp=TCP服务对象
def return_bot_version(tcp):
    T_CARTYPE = g_car_type
    T_FUNC = 0x01
    T_LEN = 0x04
    version = int(g_bot.get_version() * 10)
    if version < 0:
        version = 0
    checknum = (6 + version) % 256
    data = "$%02x%02x%02x%02x%02x#" % (T_CARTYPE, T_FUNC, T_LEN, version, checknum)
    tcp.send(data.encode(encoding="utf-8"))
    if g_debug:
        print("Rosmaster Version:", version / 10.0)
        print("tcp send:", data)

# 返回电池电压
def return_battery_voltage(tcp):
    T_CARTYPE = g_car_type
    T_FUNC = 0x02
    T_LEN = 0x04
    vol = int(g_bot.get_battery_voltage() * 10) % 256
    if vol < 0:
        vol = 0
    checknum = (T_CARTYPE + T_FUNC + T_LEN + vol) % 256
    data = "$%02x%02x%02x%02x%02x#" % (T_CARTYPE, T_FUNC, T_LEN, vol, checknum)
    tcp.send(data.encode(encoding="utf-8"))
    if g_debug:
        print("voltage:", vol / 10.0)
        print("tcp send:", data)
    return vol / 10.0

# 返回小车速度控制百分比
def return_car_speed(tcp, speed_xy, speed_z):
    T_CARTYPE = g_car_type
    T_FUNC = 0x16
    T_LEN = 0x06
    checknum = (T_CARTYPE + T_FUNC + T_LEN + int(speed_xy) + int(speed_z)) % 256
    data = "$%02x%02x%02x%02x%02x%02x#" % (T_CARTYPE, T_FUNC, T_LEN, int(speed_xy), int(speed_z), checknum)
    tcp.send(data.encode(encoding="utf-8"))
    if g_debug:
        print("speed:", speed_xy, speed_z)
        print("tcp send:", data)

# 返回小车自稳状态
def return_car_stabilize(tcp, state):
    T_CARTYPE = g_car_type
    T_FUNC = 0x17
    T_LEN = 0x04
    checknum = (T_CARTYPE + T_FUNC + T_LEN + int(state)) % 256
    data = "$%02x%02x%02x%02x%02x#" % (T_CARTYPE, T_FUNC, T_LEN, int(state), checknum)
    tcp.send(data.encode(encoding="utf-8"))
    if g_debug:
        print("stabilize:", int(state))
        print("tcp send:", data)


# 返回小车当前的XYZ速度
def return_car_current_speed(tcp):
    T_CARTYPE = g_car_type
    T_FUNC = 0x22
    T_LEN = 0x0E
    speed = g_bot.get_motion_data()
    num_x = int(speed[0]*100)
    num_y = int(speed[1]*100)
    num_z = int(speed[2]*20)
    speed_x = num_x.to_bytes(2, byteorder='little', signed=True)
    speed_y = num_y.to_bytes(2, byteorder='little', signed=True)
    speed_z = num_z.to_bytes(2, byteorder='little', signed=True)
    checknum = (T_CARTYPE + T_FUNC + T_LEN + speed_x[0] + speed_x[1] + speed_y[0] + speed_y[1] + speed_z[0] + speed_z[1]) % 256
    data = "$%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x#" % \
        (T_CARTYPE, T_FUNC, T_LEN, speed_x[0], speed_x[1], speed_y[0], speed_y[1], speed_z[0], speed_z[1], checknum)
    tcp.send(data.encode(encoding="utf-8"))
    # if g_debug:
    #     print("current_speed:", num_x, num_y, num_z)
    #     print("tcp send:", data)


# 返回阿克曼小车默认角度
def return_ackerman_angle(tcp, id, angle):
    T_CARTYPE = g_car_type
    T_FUNC = 0x50
    T_LEN = 0x06
    checknum = (T_CARTYPE + T_FUNC + T_LEN + int(id) + int(angle)) % 256
    data = "$%02x%02x%02x%02x%02x%02x#" % (T_CARTYPE, T_FUNC, T_LEN, int(id), int(angle), checknum)
    tcp.send(data.encode(encoding="utf-8"))
    if g_debug:
        print("ackerman_angle:", int(angle))
        print("tcp send:", data)

# 返回摄像头类型
def return_camera_type(tcp, type):
    T_CARTYPE = g_car_type
    T_FUNC = 0x18
    T_LEN = 0x04
    checknum = (T_CARTYPE + T_FUNC + T_LEN + int(type)) % 256
    data = "$%02x%02x%02x%02x%02x#" % (T_CARTYPE, T_FUNC, T_LEN, int(type), checknum)
    tcp.send(data.encode(encoding="utf-8"))
    if g_debug:
        print("return_camera_type:", int(type))
        print("tcp send:", data)


# 数值变换
def my_map(x, in_min, in_max, out_min, out_max):
    return (out_max - out_min) * (x - in_min) / (in_max - in_min) + out_min

# 控制X3 PLUS小车
def ctrl_car_x3plus(state):
    if state == 0:
        g_bot.set_car_run(0, g_car_stabilize_state)
    elif state == 5 or state == 6:
        speed = int(g_speed_ctrl_z * 0.7)
        g_bot.set_car_run(state, speed)
    else:
        speed = int(g_speed_ctrl_xy * 0.7)
        g_bot.set_car_run(state, speed, g_car_stabilize_state)

# 控制X3类型小车
def ctrl_car_x3(state):
    if state == 0:
        g_bot.set_car_run(0, g_car_stabilize_state)
    elif state == 5 or state == 6:
        speed = int(g_speed_ctrl_z)
        g_bot.set_car_run(state, speed)
    else:
        speed = int(g_speed_ctrl_xy)
        g_bot.set_car_run(state, speed, g_car_stabilize_state)


# 控制R2类型小车
def ctrl_car_r2(state):
    global g_akm_ctrl_state
    if state == 0:
        if g_akm_ctrl_state > 2:
            # 轮子自动回中
            g_bot.set_car_run(0, g_car_stabilize_state, 1)
            g_akm_ctrl_state = 0
        else:
            g_bot.set_car_run(0, g_car_stabilize_state)
    # elif state == 1 or state == 2:
    #     speed = int(g_speed_ctrl_xy*1.8)
    #     g_bot.set_car_run(state, speed)
    else:
        speed = int(g_speed_ctrl_xy*1.8)
        g_bot.set_car_run(state, speed, g_car_stabilize_state)
    g_akm_ctrl_state = state

# 控制X1类型小车
def ctrl_car_x1(state):
    if state == 0:
        g_bot.set_car_run(0, g_car_stabilize_state)
    elif state == 5 or state == 6:
        speed = int(g_speed_ctrl_z)
        g_bot.set_car_run(state, speed)
    else:
        speed = int(g_speed_ctrl_xy)
        g_bot.set_car_run(state, speed, g_car_stabilize_state)


# 协议解析部分
def parse_data(sk_client, data):
    # print(data)
    global g_mode, g_bot, g_car_type
    global g_akm_def_angle
    global g_motor_speed
    global g_speed_ctrl_xy, g_speed_ctrl_z
    global g_car_stabilize_state
    global g_camera_type, g_camera_state, g_camera_usb, g_camera_wide_angle
    data_size = len(data)
    # 长度校验
    if data_size < 8:
        if g_debug:
            print("The data length is too short!", data_size)
        return
    #hexa-decimal to decimal using int(xx, 16)
    if int(data[5:7], 16) != data_size-8:
        if g_debug:
            print("The data length error!", int(data[5:7], 16), data_size-8)
        return
    # 和校验
    checknum = 0
    num_checknum = int(data[data_size-3:data_size-1], 16)
    for i in range(0, data_size-4, 2):
        checknum = (int(data[1+i:3+i], 16) + checknum) % 256
        # print("check:", i, int(data[1+i:3+i], 16), checknum)
    if checknum != num_checknum:
        if g_debug:
            print("num_checknum error!", checknum, num_checknum)
            print("checksum error! cmd:0x%02x, calnum:%d, recvnum:%d" % (int(data[3:5], 16), checknum, num_checknum))
        return
    
    # 小车类型匹配
    num_carType = int(data[1:3], 16)
    if num_carType <= 0 or num_carType > 5:
        if g_debug:
            print("num_carType error!")
        return
    else:
        if g_car_type != num_carType:
            g_car_type = num_carType
            g_bot.set_car_type(g_car_type)
            if g_car_type == g_bot.CARTYPE_X3_PLUS:
                g_camera_usb = Rosmaster_Camera(video_id=0x51, debug=g_debug)
                g_camera_state = 1
            elif g_car_type == g_bot.CARTYPE_R2:
                g_camera_wide_angle = Rosmaster_Camera(video_id=0x52, debug=g_debug)
            if g_debug:
                print("set_car_type:", g_car_type)
    
    # 解析命令标记
    cmd = data[3:5]
    if cmd == "0F":  # 进入界面
        func = int(data[7:9])
        if g_debug:
            print("cmd func=", func)
        g_mode = 'Home'
        if func == 0: # 首页
            return_battery_voltage(sk_client)
        elif func == 1: # 遥控
            return_car_speed(sk_client, g_speed_ctrl_xy, g_speed_ctrl_z)
            return_car_stabilize(sk_client, g_car_stabilize_state)
            if g_car_type == g_bot.CARTYPE_X3_PLUS:
                if g_camera_type == g_camera.TYPE_USB_CAMERA:
                    return_camera_type(sk_client, 1)
                elif g_camera_type == g_camera.TYPE_DEPTH_CAMERA:
                    return_camera_type(sk_client, 2)
            g_mode = 'Standard'
        elif func == 2: # 麦克纳姆轮
            return_car_current_speed(sk_client)
            g_mode = 'MecanumWheel'

    elif cmd == "01":  # 获取硬件版本号
        if g_debug:
            print("get version")
        return_bot_version(sk_client)

    elif cmd == "02":  # 获取电池电压
        if g_debug:
            print("get voltage")
        return_battery_voltage(sk_client)

    elif cmd == "10":  # 控制小车
        num_x = int(data[7:9], 16)
        num_y = int(data[9:11], 16)
        if num_x > 127:
            num_x = num_x - 256
        if num_y > 127:
            num_y = num_y - 256
        speed_x = num_y / 100.0
        speed_y = -num_x / 100.0
        if speed_x == 0 and speed_y == 0:
            if g_car_type == g_bot.CARTYPE_R2:
                g_bot.set_car_run(0, g_car_stabilize_state, 1)
            else:
                g_bot.set_car_run(0, g_car_stabilize_state)
        else:
            if g_car_type == g_bot.CARTYPE_R2:
                speed_y = my_map(speed_y, -1, 1, AKM_LIMIT_ANGLE/-1000.0, AKM_LIMIT_ANGLE/1000.0)
                # speed_z = my_map(speed_y, -1, 1, 3.0, -3.0)
                g_bot.set_car_motion(speed_x*1.8, speed_y, 0)
            else:
                g_bot.set_car_motion(speed_x, speed_y, 0)
        if g_debug:
            print("speed_x:%.2f, speed_y:%.2f" % (speed_x, speed_y))


    # 控制PWM舵机
    elif cmd == "11":
        num_id = int(data[7:9], 16)
        num_angle = int(data[9:11], 16)
        if g_debug:
            print("pwm servo id:%d, angle:%d" % (num_id, num_angle))
        if g_car_type == g_bot.CARTYPE_R2:
            angle_mini = (AKM_DEFAULT_ANGLE - AKM_LIMIT_ANGLE)
            angle_max = (AKM_DEFAULT_ANGLE + AKM_LIMIT_ANGLE)
            if num_angle < angle_mini:
                num_angle = angle_mini
            elif num_angle > angle_max:
                num_angle = angle_max
            g_bot.set_pwm_servo(num_id, num_angle)
        else:
            g_bot.set_pwm_servo(num_id, num_angle)


    # 控制机械臂
    elif cmd == "12":
        num_id = int(data[7:9], 16)
        num_angle_l = int(data[9:11], 16)
        num_angle_h = int(data[11:13], 16)
        uart_servo_angle = num_angle_h * 256 + num_angle_l
        if g_debug:
            print("uart servo id:%d, angle:%d" % (num_id, uart_servo_angle))
        if 1 < num_id < 5:
            uart_servo_angle = 180 - uart_servo_angle
        g_bot.set_uart_servo_angle(num_id, uart_servo_angle)


    # 设置蜂鸣器
    elif cmd == "13":
        num_state = int(data[7:9], 16)
        num_delay = int(data[9:11], 16)
        if g_debug:
            print("beep:%d, delay:%d" % (num_state, num_delay))
        delay_ms = 0
        if num_state > 0:
            if num_delay == 255:
                delay_ms = 1
            else:
                delay_ms = num_delay * 10
        g_bot.set_beep(delay_ms)

    # 读取机械臂舵机角度。
    elif cmd == "14":
        num_id = int(data[7:9], 16)
        if g_debug:
            print("read angle:%d" % num_id)
        if num_id == 6:
            return_arm_angle(sk_client)
    
    # 按键控制
    elif cmd == "15":
        num_dir = int(data[7:9], 16)
        if g_debug:
            print("btn ctl:%d" % num_dir)
        if g_car_type == g_bot.CARTYPE_X3_PLUS:
            ctrl_car_x3plus(num_dir)
        elif g_car_type == g_bot.CARTYPE_R2:
            ctrl_car_r2(num_dir)
        elif g_car_type == g_bot.CARTYPE_X3:
            ctrl_car_x3(num_dir)
        elif g_car_type == g_bot.CARTYPE_X1:
            ctrl_car_x1(num_dir)
        else:    
            print("car type error!")
        if g_debug:
            print("car ctrl:", num_dir)
    
    # 控制速度
    elif cmd == '16':
        num_speed_xy = int(data[7:9], 16)
        num_speed_z = int(data[9:11], 16)
        if g_debug:
            print("speed ctl:%d, %d" % (num_speed_xy, num_speed_z))
        g_speed_ctrl_xy = num_speed_xy
        g_speed_ctrl_z = num_speed_z
        if g_speed_ctrl_xy > 100:
            g_speed_ctrl_xy = 100
        if g_speed_ctrl_xy < 0:
            g_speed_ctrl_xy = 0
        if g_speed_ctrl_z > 100:
            g_speed_ctrl_z = 100
        if g_speed_ctrl_z < 0:
            g_speed_ctrl_z = 0

    # 自稳开关
    elif cmd == '17':
        num_stab = int(data[7:9], 16)
        if g_debug:
            print("car stabilize:%d" % num_stab)
        if num_stab > 0:
            g_car_stabilize_state = 1
        else:
            g_car_stabilize_state = 0

    # X3PLUS, 摄像头切换
    elif cmd == '18':
        num_camera = int(data[7:9], 16)
        if g_debug:
            print("select camera:%d" % num_camera)
        if num_camera == 1:
            g_camera_type = g_camera.TYPE_USB_CAMERA
        elif num_camera == 2:
            g_camera_type = g_camera.TYPE_DEPTH_CAMERA


    # 麦克纳姆轮控制
    elif cmd == '20':
        num_id = int(data[7:9], 16)
        num_speed = int(data[9:11], 16)
        if num_speed > 127:
            num_speed = num_speed - 256
        if g_debug:
            print("mecanum wheel ctrl:%d, %d" % (num_id, num_speed))
        if num_id >= 0 and num_id <= 4:
            if num_speed > 100:
                num_speed = 100
            if num_speed < -100:
                num_speed = -100
            if num_id == 0:
                g_motor_speed[0] = 0
                g_motor_speed[1] = 0
                g_motor_speed[2] = 0
                g_motor_speed[3] = 0
            else:
                g_motor_speed[num_id-1] = num_speed
            g_bot.set_motor(g_motor_speed[0], g_motor_speed[1], g_motor_speed[2], g_motor_speed[3])

    # 更新速度
    elif cmd == '21':
        num_speed_m1 = int(data[7:9], 16)
        num_speed_m2 = int(data[9:11], 16)
        num_speed_m3 = int(data[11:13], 16)
        num_speed_m4 = int(data[13:15], 16)
        if num_speed_m1 > 127:
            num_speed_m1 = num_speed_m1 - 256
        if num_speed_m2 > 127:
            num_speed_m2 = num_speed_m2 - 256
        if num_speed_m3 > 127:
            num_speed_m3 = num_speed_m3 - 256
        if num_speed_m4 > 127:
            num_speed_m4 = num_speed_m4 - 256
        if g_debug:
            print("mecanum wheel update:%d, %d, %d, %d" % (num_speed_m1, num_speed_m2, num_speed_m3, num_speed_m4))
        g_motor_speed[0] = num_speed_m1
        g_motor_speed[1] = num_speed_m2
        g_motor_speed[2] = num_speed_m3
        g_motor_speed[3] = num_speed_m4
        g_bot.set_motor(g_motor_speed[0], g_motor_speed[1], g_motor_speed[2], g_motor_speed[3])

    # 设置彩色灯带的颜色
    elif cmd == "30":
        num_id = int(data[7:9], 16)
        num_r = int(data[9:11], 16)
        num_g = int(data[11:13], 16)
        num_b = int(data[13:15], 16)
        if g_debug:
            print("lamp:%d, r:%d, g:%d, b:%d" % (num_id, num_r, num_g, num_b))
        g_bot.set_colorful_lamps(num_id, num_r, num_g, num_b)

    # 设置彩色灯带的特效
    elif cmd == "31":
        num_effect = int(data[7:9], 16)
        num_speed = int(data[9:11], 16)
        if g_debug:
            print("effect:%d, speed:%d" % (num_effect, num_speed))
        g_bot.set_colorful_effect(num_effect, num_speed, 255)

    # 设置彩色灯带的单色呼吸灯效果的颜色
    elif cmd == "32":
        num_color = int(data[7:9], 16)
        if g_debug:
            print("breath color:%d" % num_color)
        if num_color == 0:
            g_bot.set_colorful_effect(0, 255, 255)
        else:
            g_bot.set_colorful_effect(3, 255, num_color - 1)


    # 机械臂中位校准
    elif cmd == '40':
        num_cali = int(data[7:9], 16)
        if g_debug:
            print("arm offset:%d" % num_cali)
        if num_cali == 1:
            for i in range(6):
                id = int(i+1)
                state = g_bot.set_uart_servo_offset(id)
                return_arm_offset_state(sk_client, id, state)
            time.sleep(.01)
            g_bot.set_uart_servo_torque(True)

    # 机械臂归中
    elif cmd == '41':
        num_verify = int(data[7:9], 16)
        if g_debug:
            print("arm up:%d" % num_verify)
        if num_verify == 1:
            g_bot.set_uart_servo_torque(True)
            time.sleep(.01)
            angle_array = [90, 90, 90, 90, 90, 180]
            g_bot.set_uart_servo_angle_array(angle_array)
            time.sleep(.5)

    # 机械臂扭矩开关
    elif cmd == '42':
        num_verify = int(data[7:9], 16)
        if g_debug:
            print("arm torque:%d" % num_verify)
        if num_verify == 0:
            g_bot.set_uart_servo_torque(False)
        else:
            g_bot.set_uart_servo_torque(True)

    # 机械臂姿态
    elif cmd == '43':
        num_pose = int(data[7:9], 16)
        if g_debug:
            print("arm pose:%d" % num_pose)
        if num_pose == 1: # 防撞姿态
            angle_array = [90, 180-0, 180-180, 180-180, 90, 30]
            g_bot.set_uart_servo_angle_array(angle_array)
            time.sleep(.2)
        elif num_pose == 2: # 跳舞
            task_dance = threading.Thread(target = task_dance_handle)
            task_dance.setDaemon(True)
            task_dance.start()
        elif num_pose == 3: # 巡线姿态
            angle_array = [90, 180-40, 180-180, 180-180, 90, 30]
            g_bot.set_uart_servo_angle_array(angle_array)
            time.sleep(.2)


    # 读取阿克曼小车默认角度
    elif cmd == '50':
        num_id = int(data[7:9], 16)
        if g_debug:
            print("akm angle read:%d" % num_id)
        if num_id == AKM_PWMSERVO_ID:
            g_akm_def_angle = g_bot.get_akm_default_angle()
            return_ackerman_angle(sk_client, AKM_PWMSERVO_ID, g_akm_def_angle)

    # 临时修改阿克曼舵机的默认角度
    elif cmd == '51':
        num_id = int(data[7:9], 16)
        num_angle = int(data[9:11], 16)
        if g_debug:
            print("akm change angle:%d, %d" % (num_id, num_angle))
        if 60 <= num_angle <= 120 and num_id == AKM_PWMSERVO_ID:
            g_akm_def_angle = num_angle
            g_bot.set_akm_default_angle(num_angle)

    # 确认修改阿克曼小车默认角度
    elif cmd == '52':
        num_verify = int(data[7:9], 16)
        if g_debug:
            print("akm save angle:%d, %d" % (num_verify, g_akm_def_angle))
        if num_verify == 1:
            g_bot.set_akm_default_angle(g_akm_def_angle, True)
            time.sleep(.1)

    # 控制阿克曼小车前轮舵机角度
    elif cmd == '53':
        num_id = int(data[7:9], 16)
        num_angle = int(data[9:11], 16)
        if num_angle > 127:
            num_angle = num_angle - 256
        if g_debug:
            print("akm angle:%d, %d" % (num_id, num_angle))
        if -45 <= num_angle <= 45 and num_id == AKM_PWMSERVO_ID:
            g_bot.set_akm_steering_angle(num_angle)


if __name__ == "__main__":
    #test serial port
    #in terminal: ls -l /dev/ttyUSB0
    #screen /dev/ttyUSB0 115200
    com="/dev/ttyUSB0"
    #ser = serial.Serial(com, 115200)
    g_car_type = 0 #1
    g_debug= True
    g_bot = Rosmaster(car_type=g_car_type, com=com, debug=g_debug)
    # Help can print all the bot methods and remarks
    help(g_bot)
    g_bot.create_receive_threading()

    # 速度控制
    g_speed_ctrl_xy = 100
    g_speed_ctrl_z = 100
    g_motor_speed = [0, 0, 0, 0]
    g_car_stabilize_state = 0


    # 阿克曼小车参数
    AKM_DEFAULT_ANGLE = 90
    AKM_LIMIT_ANGLE = 45
    AKM_PWMSERVO_ID = 1

    g_akm_def_angle = 90
    g_akm_ctrl_state = 0

    # TCP未接收命令超时计数
    g_tcp_except_count = 0

    g_motor_speed = [0, 0, 0, 0]