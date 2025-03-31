
import rtde_receive
import rtde_control 



#define UR3e
rtde_c = rtde_control.RTDEControlInterface("192.168.1.64")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.64")



while True:
    dio = rtde_r.getActualDigitalInputBits()
    if dio > 0:
        print(bin(dio)[2])

