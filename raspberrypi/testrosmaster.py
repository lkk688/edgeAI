from Rosmaster_Lib import Rosmaster 
import serial

if __name__ == "__main__":
    #test serial port
    #in terminal: ls -l /dev/ttyUSB0
    #screen /dev/ttyUSB0 115200
    com="/dev/ttyUSB0"
    #ser = serial.Serial(com, 115200)
    bot = Rosmaster(car_type=1, com=com)
    # Help can print all the bot methods and remarks
    help(bot)