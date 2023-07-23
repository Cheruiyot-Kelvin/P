from socket import * 
import optparse      
import threading     

def portScan(host, port):
    try:
        s = socket(AF_INET, SOCK_STREAM) 
        s.connect((host, int(port)))     
        print(host + " tcp/" + str(port) + " open")
        s.close()
        
    except:
        print(host + " tcp/" + str(port) + " closed") 
        
def main():
    
    parser = optparse.OptionParser("uasge%prog " + "-H <specify target host> -p <specify ports separated by ',' > ")
    parser.add_option("-H" , '--host' , dest = 'targethost' , type = 'string' , help = 'specify target hsot')
    parser.add_option("-p", "--ports", dest = 'targetports', type = 'string', help = 'specify target ports separated by "," ')
    
    option ,  args = parser.parse_args() 
    
    thost = option.targethost 
    tports = str(option.targetports).split(",")
    
    if thost ==  None or tports[0] == None:
        print(parser.usage)       
        exit(0) 
    
    
    setdefaulttimeout(1)           
    host_ip = gethostbyname(thost)
    
    for port in tports:
        t = threading.Thread(target=portScan, args=(thost,port)) 
        t.start()
        
main()
