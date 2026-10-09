from dynamixel_sdk import *
import numpy as np

""" This class should interface the control of the motors using the Dynamixel SDK."""
class Motors:
    # Class variables
    POS_CONTROL = 3
    VEL_CONTROL = 1
    CUR_CONTROL = 0
    CONTROL_DICT = {"pos": POS_CONTROL,
                    "vel": VEL_CONTROL,
                    "cur": CUR_CONTROL}
    
    """ Check which COM-port the motor is connected to by using the command python "-m serial.tools.list_ports" in the terminal."""
    def __init__(self, port = "COM3", baudrate = 3000000):
        # Set memeber variables
        self.port = port
        self.baudrate = baudrate
        self.portHandler = PortHandler(self.port)
        self.packetHandler = PacketHandler(2.0)
        self.ping_num = 50
        self.RPM = 0.229 * 2 * np.pi / 60   # 1 velocity unit = 0.229 rpm * conversion to rad/s
        self.motor_ids = []
        self.num_motors = 0
        self.torque_enabled = np.zeros(0, dtype=bool)
        self._closed = False
        
        # Open port and set baudrate
        if self.portHandler.openPort():
            print("Succeeded to open the port")
        else:
            self.close()
            raise Exception("Failed to open the port")
        if self.portHandler.setBaudRate(self.baudrate):
            print("Succeeded to change the baudrate")
        else:
            self.close()
            raise Exception("Failed to set the baudrate")
        
        # Get motor IDs
        self.motor_ids = self.get_motor_ids()
        self.num_motors = len(self.motor_ids)
        self.torque_enabled = np.zeros(self.num_motors, dtype=bool)
        if self.num_motors == 0:
            self.close()
            raise Exception("No motors found")
        print(f"Found {self.num_motors} motors with IDs: {self.motor_ids}")

        # Set up groupsyncreads and writes if there is more than one motor.
        if self.num_motors > 1:
            self.pos_gsr = GroupSyncRead(self.portHandler, self.packetHandler, 132, 4)
            self.vel_gsr = GroupSyncRead(self.portHandler, self.packetHandler, 128, 4)
            self.cur_gsr = GroupSyncRead(self.portHandler, self.packetHandler, 126, 2)
            for i in range(self.num_motors):
                dxl_addparam_result = self.pos_gsr.addParam(self.motor_ids[i])
                if dxl_addparam_result != True:
                    self.close()
                    raise Exception("Failed to add motor ID to group sync read for position")
                dxl_addparam_result = self.vel_gsr.addParam(self.motor_ids[i])
                if dxl_addparam_result != True:
                    self.close()
                    raise Exception("Failed to add motor ID to group sync read for velocity")
                dxl_addparam_result = self.cur_gsr.addParam(self.motor_ids[i])
                if dxl_addparam_result != True:
                    self.close()
                    raise Exception("Failed to add motor ID to group sync read for current")
                
            self.pos_gsw = GroupSyncWrite(self.portHandler, self.packetHandler, 116, 4)
            self.vel_gsw = GroupSyncWrite(self.portHandler, self.packetHandler, 104, 4)
            self.cur_gsw = GroupSyncWrite(self.portHandler, self.packetHandler, 102, 2)

        # Set initial motor parameters
        self.control_mode = np.zeros(self.num_motors, dtype = int) # 0: current, 1: velocity, 3: position

        self.minPos = np.zeros(self.num_motors)
        self.maxPos = np.zeros(self.num_motors)
        self.minVel = np.zeros(self.num_motors)
        self.maxVel = np.zeros(self.num_motors)
        self.minCur = np.zeros(self.num_motors)
        self.maxCur = np.zeros(self.num_motors)

        # Initialize the arrays that will contain current positions, velocities, currents and torques
        self.now_pos = np.zeros(self.num_motors)
        self.now_vel = np.zeros(self.num_motors)
        self.now_cur = np.zeros(self.num_motors)
        self.now_tor = np.zeros(self.num_motors)

        # Set syncronous control parameters
        self.control_positions = np.zeros(self.num_motors)
        self.control_velocities = np.zeros(self.num_motors)
        self.control_currents = np.zeros(self.num_motors)

        # initialize all motors to position control mode
        self.get_motor_params()
        print(f"Max and min positions: {self.maxPos}, {self.minPos}, current: {self.maxCur}, {self.minCur}, velocity: {self.maxVel}, {self.minVel}")

        # self.set_cont_mode("pos")
        self.set_cont_mode("cur")

        self.enable_torque()

    """ Enable torque for all motors """
    def enable_torque(self):
        for i in range(self.num_motors):
            if not self.torque_enabled[i]:
                self.write(self.motor_ids[i], 64, 1, 1)
                self.torque_enabled[i] = True
            else:
                print(f"Motor {self.motor_ids[i]} torque is already enabled.")
        print("All motors torque enabled.")

    """ Disable torque for all motors """
    def disable_torque(self):
        for i in range(self.num_motors):
            if self.torque_enabled[i]:
                self.write(self.motor_ids[i], 64, 1, 0)
                self.torque_enabled[i] = False
            else:
                print(f"Motor {self.motor_ids[i]} torque is already disabled.")
        print("All motors torque disabled.")

    """ Ping all possible motor IDs and return a list of the IDs that respond. """
    def get_motor_ids(self):
        motor_ids = []
        for i in range(self.ping_num):
            dxl_model_number, dxl_comm_result, dxl_error = self.packetHandler.ping(self.portHandler, i)
            if dxl_comm_result != COMM_SUCCESS:
                n=0
            elif dxl_error != 0:
                n=0
            else:
                motor_ids.append(i)
        return motor_ids

    """ Read motor parameters such as control mode, position, velocity and current limits from each motor."""
    def get_motor_params(self):
        for i in range(self.num_motors):
            self.control_mode[i] = self.read(self.motor_ids[i], 11, 1)
            self.maxCur[i] = self.read(self.motor_ids[i], 38, 2)
            self.minCur[i] = -self.maxCur[i]
            self.minPos[i] = self.read(self.motor_ids[i], 52, 4)
            self.maxPos[i] = self.read(self.motor_ids[i], 48, 4)
            self.maxVel[i] = self.read(self.motor_ids[i], 44, 4)
            self.minVel[i] = -self.maxVel[i]
            print("Motor ID: {}, Control Mode: {}, MinPos: {}, MaxPos: {}, MinVel: {}, MaxVel: {}, MinCur: {}, MaxCur: {}".format(
                self.motor_ids[i], self.control_mode[i], self.minPos[i], self.maxPos[i], self.minVel[i], self.maxVel[i], self.minCur[i], self.maxCur[i]))
        print("Motor parameters read successfully.")
    
    # Single motor read and write functions
    """ Write commands to the motor. motor_id: ID of the motor, add_write: address to write to, byte_num: number of bytes to write (1, 2 or 4), comm: command to write"""
    def write(self, motor_id, add_write, byte_num, comm):
        assert (byte_num in [1,2,4]), "the writting byte should be one of [1, 2, 4]"
        comm = int(comm)
        if (byte_num == 1):
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, motor_id, add_write, comm)
        elif (byte_num == 2):
            dxl_comm_result, dxl_error = self.packetHandler.write2ByteTxRx(self.portHandler, motor_id, add_write, comm)
        else:
            dxl_comm_result, dxl_error = self.packetHandler.write4ByteTxRx(self.portHandler, motor_id, add_write, comm)
        
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
    
    """ Read data from the motor. motor_id: ID of the motor, add_read: address to read from, byte_num: number of bytes to read (1, 2 or 4) """
    def read(self, motor_id, add_read, byte_num):
        assert (byte_num in [1,2,4]), "the reading byte should be one of [1, 2, 4]"
        if (byte_num == 1):
            cl_dxl, cl_dxl_comm_result, cl_dxl_error = self.packetHandler.read1ByteTxRx(self.portHandler, motor_id, add_read)
        elif (byte_num == 2):
            cl_dxl, cl_dxl_comm_result, cl_dxl_error = self.packetHandler.read2ByteTxRx(self.portHandler, motor_id, add_read)
        else:
            cl_dxl, cl_dxl_comm_result, cl_dxl_error = self.packetHandler.read4ByteTxRx(self.portHandler, motor_id, add_read)

        if cl_dxl_comm_result != COMM_SUCCESS:
            msg = self.packetHandler.getTxRxResult(cl_dxl_comm_result)
            print("%s" % msg)
            raise RuntimeError(f"Read error (motor {motor_id}, addr {add_read}): {msg}")
        elif cl_dxl_error != 0:
            msg = self.packetHandler.getRxPacketError(cl_dxl_error)
            print("%s" % msg)
            raise RuntimeError(f"Motor error (motor {motor_id}, addr {add_read}): {msg}")
        else:
            return cl_dxl

    # TODO: Synconous read and write functions for multiple motors.
    """ Write commands syncronosly to the motors. gsw: GroupSyncWrite object, motor_id: ID of the motor, byte_num: number of bytes to write (2 or 4), goal: command to write """
    def sync_write(self, gsw, motor_id, byte_num, goal):
        assert (byte_num in [2,4]), "the writing byte should be one of [2, 4]"
        goal = int(goal)
        if byte_num == 4:
            goal_var = [DXL_LOBYTE(DXL_LOWORD(goal)), DXL_HIBYTE(DXL_LOWORD(goal)), DXL_LOBYTE(DXL_HIWORD(goal)), DXL_HIBYTE(DXL_HIWORD(goal))]
        else:
            goal_var = [DXL_LOBYTE(goal), DXL_HIBYTE(goal)]
        dxl_addparam_result = gsw.addParam(motor_id, goal_var)
        if dxl_addparam_result != True:
            self.close()
            raise Exception("Failed to add motor ID to group sync write")

    """ Read data syncronosly from all motors. The read values are stored in now_pos, now_vel, now_cur and now_tor """
    def sync_read(self):
        assert (self.num_motors > 1), "sync read is only available for more than one motor"
        # syncread
        dxl_comm_result = self.pos_gsr.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        dxl_comm_result = self.vel_gsr.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        dxl_comm_result = self.cur_gsr.txRxPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        
        for i in range(self.num_motors):
            dxl_getdata_result = self.pos_gsr.isAvailable(self.motor_ids[i], 132, 4)
            if dxl_getdata_result != True:
                self.close()
                raise Exception("Failed to get data from motor ID for position")
            dxl_getdata_result = self.vel_gsr.isAvailable(self.motor_ids[i], 128, 4)
            if dxl_getdata_result != True:
                self.close()
                raise Exception("Failed to get data from motor ID for velocity")
            dxl_getdata_result = self.cur_gsr.isAvailable(self.motor_ids[i], 126, 2)
            if dxl_getdata_result != True:
                self.close()
                raise Exception("Failed to get data from motor ID for current")
            
            self.now_pos[i] = self.pos_gsr.getData(self.motor_ids[i], 132, 4)
            self.now_vel[i] = self.byte2num(self.vel_gsr.getData(self.motor_ids[i], 128, 4), 4) * self.RPM
            self.now_cur[i] = self.byte2num(self.cur_gsr.getData(self.motor_ids[i], 126, 2), 2) * 2.69 * 0.001   # 1 current unit = 2.69 mA * conversion to A
            self.now_tor[i] = self.cur2torq(self.now_cur[i])

    """ Send syncronous control commands to all motors. byte_num: number of bytes to write (2 or 4), control_commands: list of commands for each motor """
    def sync_motor_control(self, byte_num, control_commands):
        assert (byte_num in [2,4]), "the writing byte should be one of [2, 4]"
        assert (self.num_motors > 1), "sync control is only available for more than one motor"
        assert (len(control_commands) == self.num_motors), "the length of control_commands should be equal to the number of motors"
        for i in range(self.num_motors):
            if self.control_mode[i] == self.POS_CONTROL:
                self.sync_write(self.pos_gsw, self.motor_ids[i], byte_num, control_commands[i])
            elif self.control_mode[i] == self.VEL_CONTROL:
                self.sync_write(self.vel_gsw, self.motor_ids[i], byte_num, control_commands[i])
            elif self.control_mode[i] == self.CUR_CONTROL:
                self.sync_write(self.cur_gsw, self.motor_ids[i], byte_num, control_commands[i])
            else:
                self.close()
                raise Exception(f"Motor {self.motor_ids[i]} has unknown control mode.")
    
        # syncwrite
        if self.control_mode[0] == self.POS_CONTROL:
            dxl_comm_result = self.pos_gsw.txPacket()
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            self.pos_gsw.clearParam()
        elif self.control_mode[0] == self.VEL_CONTROL:
            dxl_comm_result = self.vel_gsw.txPacket()
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            self.vel_gsw.clearParam()
        elif self.control_mode[0] == self.CUR_CONTROL:
            dxl_comm_result = self.cur_gsw.txPacket()
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            self.cur_gsw.clearParam()

    """ Set control mode for all motors. mode: "pos", "vel" or "cur" """
    def set_cont_mode(self, mode = "pos"):
        for i in range(self.num_motors):
            if self.control_mode[i] != self.CONTROL_DICT[mode]:
                self.write(self.motor_ids[i], 11, 1, self.CONTROL_DICT[mode])
                self.control_mode[i] = self.CONTROL_DICT[mode]
        print(f"All motors set to {mode} control mode.")

    """ Get the current position of all motors in ticks (0-4095) """
    def get_position(self, motor_id=None):
        if motor_id is None:
            for i in range(self.num_motors):
                self.now_pos[i] = self.read(self.motor_ids[i], 132, 4)
            return self.now_pos
        else:
            assert motor_id in self.motor_ids, f"Motor ID {motor_id} not found."
            idx = self.motor_ids.index(motor_id)
            pos = self.read(motor_id, 132, 4)
            self.now_pos[idx] = pos
            return pos
    
    """ Get the current velocity of all motors in rad/s """
    def get_velocity(self, motor_id=None):
        if motor_id is None:
            for i in range(self.num_motors):
                vel = self.read(self.motor_ids[i], 128, 4)
                vel = self.byte2num(vel, 4)
                self.now_vel[i] = vel * self.RPM
            return self.now_vel
        else:
            assert motor_id in self.motor_ids
            vel = self.read(motor_id, 128, 4)
            vel = self.byte2num(vel, 4)
            vel = vel * self.RPM
            self.now_vel[self.motor_ids.index(motor_id)] = vel
            return vel

    """ Get the current current of all motors in mA """
    def get_current_torque(self, motor_id=None):
        if motor_id is None:
            for i in range(self.num_motors):
                cur = self.read(self.motor_ids[i], 126, 2)
                cur = self.byte2num(cur, 2)
                self.now_cur[i] = cur * 2.69 * 0.001   # 1 current unit = 2.69 mA * conversion to A
                # Convert current to torque
                self.now_tor[i] = self.cur2torq(self.now_cur[i])
            return self.now_cur, self.now_tor
        else:
            assert (motor_id in self.motor_ids), f"Motor ID {motor_id} not found."
            cur = self.read(motor_id, 126, 2)
            cur = self.byte2num(cur, 2)
            cur = cur * 2.69 * 0.001   # 1 current unit = 2.69 mA * conversion to A
            self.now_cur[self.motor_ids.index(motor_id)] = cur
            tor = self.cur2torq(cur)
            self.now_tor[self.motor_ids.index(motor_id)] = tor
            return cur, tor
    
    """ Control the a given motor (motor_id), using the given control method. Command should be the desired position, velocity or current based on the control mode. Remember to use correct conversion functions in order to have values the motor understands. """
    def sendMotorCommand(self, motor_id, command):
        assert (motor_id in self.motor_ids), f"Motor ID {motor_id} not found."
        idx = self.motor_ids.index(motor_id)
        if self.control_mode[idx] == self.POS_CONTROL:
            self.write(motor_id, 116, 4, command)
        elif self.control_mode[idx] == self.VEL_CONTROL:
            self.write(motor_id, 104, 4, command)
        elif self.control_mode[idx] == self.CUR_CONTROL:
            self.write(motor_id, 102, 2, command)
        else:
            self.close()
            raise Exception(f"Motor {motor_id} has unknown control mode.")

    def get_feedback(self):
        if self.num_motors > 1:
            self.sync_read()
        else:
            self.get_position()
            self.get_velocity()
            self.get_current_torque()

    """ Close the port """
    def close(self):
        if not self._closed:
            self.disable_torque()
            self.portHandler.closePort()
            self._closed = True
            print("Port closed.")

    """ Change the baudrate of all motors new_baud: 0 = 9,600, 1 = 57,600, 2 = 115,200, 3 = 1,000,000, 4 = 2,000,000, 5 = 3,000,000, 6 = 4,000,000, 7 = 4,500,000 """
    def changeBaudrate(self, new_baud=7):
        self.disable_torque()
        for i in range(self.num_motors):
            self.write(self.motor_ids[i], 8, 1, new_baud)
        self.close()

    # Conversion methods
    """ Convert unsigned int to signed int based on the number of bytes (2 or 4) """
    def byte2num(self, var, byte_num):
        rvar = 0.00
        if byte_num == 4:
            if var > 0x7fffffff:
                rvar = var - 0xffffffff - 1
            else:
                rvar = var
        elif byte_num == 2:
            if var > 0x7fff:
                rvar = var - 0xffff - 1
            else:
                rvar = var
        else:
            rvar = var
        return rvar

    """ Convert current (A) to torque (Nm). This function should be modified based on the motor specifications. """
    def cur2torq(self, current):
        torque = current / (1.38 / 2.38)
        return torque
    
    """ Convert torque (Nm) to current control. This function should be modified based on the motor specifications. """
    def torq2curcom(self, torque):
        current = torque * (1.38 / 2.38)
        current = round(current/(2.69 * 0.001))
        return current
    
    """ Convert current control to torque """
    def curcom2torq(self, current):
        return current * 2.69 * 0.001 / (1.38 / 2.38)
    
    """ Convert position to angle in radians """
    def pos2angle(self, position, maxPos, minPos):
        return 2* np.pi * (position - minPos) / (maxPos - minPos)
    
    """ Convert angle in radians to position """
    def angle2pos(self, angle, maxPos, minPos):
        return minPos + (maxPos - minPos) * angle / (2 * np.pi)
