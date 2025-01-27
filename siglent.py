import time
import pyvisa
from scipy.interpolate import interp1d


def measV(DMM, acdc):
    cmd1 = 'MEAS:VOLT:%s? AUTO' %acdc
    delay = 0.01  # delay in seconds (50 ms)
    time.sleep(delay)

    # take in value as a string
    resultString = str(DMM.query(cmd1))

    # check if first character is a '+' or a '-'
    if(resultString[0] == '+'):
        boolSign = True
    elif(resultString[0] == '-'):
        boolSign = False

    # extract whole number with decimals
    if(boolSign == True):
        result = float(resultString.split("+",1)[1].split("E",1)[0])
    elif(boolSign == False):
        result = float(resultString.split("-",1)[1].split("E",1)[0])

    # determine exponent
    resultExp = float(resultString.split("E",1)[1])

    # apply power to whole number, store this number in a variable
    result = result*pow(10,resultExp)

    # if '+' then value = value
    # if '-' then value = -value
    if(boolSign == True):
        result = result
    elif(boolSign == False):
        result = -result

    # return value
    # print(str(result) + ' V')
    return result


def measI(DMM, acdc):
    cmd1 = 'MEAS:CURR:%s? AUTO' %acdc
    delay = 0.01  # delay in seconds (50 ms)
    time.sleep(delay)
    # take in value as a string
    resultString = str(DMM.query(cmd1))

    # check if first character is a '+' or a '-'
    if(resultString[0] == '+'):
        boolSign = True
    elif(resultString[0] == '-'):
        boolSign = False

    # extract whole number with decimals
    if(boolSign == True):
        result = float(resultString.split("+",1)[1].split("E",1)[0])
    elif(boolSign == False):
        result = float(resultString.split("-",1)[1].split("E",1)[0])

    # determine exponent
    resultExp = float(resultString.split("E",1)[1])

    # apply power to whole number, store this number in a variable
    result = result*pow(10,resultExp)

    # if '+' then value = value
    # if '-' then value = -value
    if(boolSign == True):
        result = result
    elif(boolSign == False):
        result = -result

    # return value
    # print(str(result) + ' A')
    return result


# Function to set voltage on the power supply
def set_voltage(ps, voltage):
    # SCPI command to set voltage
    ps.write(f"VOLT {voltage}")


def set_output(ps, state):
    # SCPI command to turn on output
    ps.write(f"OUTP CH1,{state}")

def read_current(ps):
    # SCPI command to read current
    current = ps.query(f"MEASure:CURRent?", delay=0.01)


    return current


if __name__ == "__main__":
    rm = pyvisa.ResourceManager()
    print(rm.list_resources())
    # DMM_v = rm.open_resource('USB0::0xF4EC::0xEE38::SDM35FAC4R0253::INSTR')  # Digital Multimeter
    # DMM_i = rm.open_resource('USB0::0xF4EC::0x1201::SDM35HBQ803105::INSTR')  # Digital Multimeter
    PS = rm.open_resource('USB0::0xF4EC::0x1410::SPD13DCC4R0058::INSTR')  # Power Supply
    PS.write_termination = '\n'
    PS.read_termination = '\n'
    # Set the voltage
    time.sleep(0.04)
    set_output(PS, state='ON')
    set_voltage(PS, voltage=0.5)
    time.sleep(1)
    print(float(read_current(PS)))
    set_output(PS, state='OFF')
    time.sleep(1)
    PS.close()
    print('DONE')



