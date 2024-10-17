import pygame
import math

# print("Enter Fighter initial position x and y")
# fighterPos = int(input()) , int(input())
# print("Enter Velocity of Figher plane")
# VF = int(input())

fighterPos = (0,0)
VF = 20
T = 0
width , height = 1000 , 600

bomberPlane = []
fighterPlane = [fighterPos]

for line in open("./Bomber_Path.txt"):
    t , x , y = line.strip().split(",")
    t = int(t); x = int(x); y = int(y)
    T = max(T , t)
    bomberPlane.append((x,y))


def calculateDistance(bomberPos , fighterPos):
    y = pow(bomberPos[1]-fighterPos[1],2)
    x = pow(bomberPos[0] - fighterPos[0],2)
    return math.sqrt(y + x)

def coord(pos):
    "Convert world coordinates to pixel coordinates."
    return (3 * (pos[0] + 50), 6 * (pos[1] + 50))

pygame.init()
pygame.display.set_caption("Pure Pursuit")

width , height = 1000 , 600
screenSize = (width,height)
screen = pygame.display.set_mode(screenSize)

f = pygame.font.get_fonts()[0]
font = pygame.font.SysFont(f, 32)

position_boomber = font.render("B", True, (255,0,0), (0,0,0))
position_fighter = font.render("F", True, (0,255,0), (0,0,0))
position_match = font.render("Caught", True, (0,255,0), (0,0,0))
position_escape = font.render("Escaped", True, (255,0,0), (0,0,0))


textRect1 = position_boomber.get_rect()
textRect2 = position_fighter.get_rect()
textRect4 = position_match.get_rect()
textRect5 = position_escape.get_rect()


running = True
t = 0

while running:
    screen.fill((0,0,0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    while t  <= T and running:
        pygame.time.delay(900)
        position_time = font.render("Time:"+str(t), True, (255,255,255), (0,0,0))
        textRect3 = position_time.get_rect()

        textRect3.center = (90,height-50)
        screen.blit(position_time , textRect3)

        if t == 11:
            textRect5.center = (width / 2 , height / 2)
            screen.blit(position_escape , textRect5)
            print("Boomber Escaped!!")
            running = False

        if t > 0:
            pygame.draw.line(screen , (0,255,0) , coord(fighterPlane[t]) ,coord(fighterPlane[t-1]), 2)
            pygame.draw.line(screen , (255,0,0) , coord(bomberPlane[t]) ,coord(bomberPlane[t-1]), 2)
            pygame.draw.circle(screen , (255,255,255) , coord(fighterPlane[t]) , 4)
            pygame.draw.circle(screen , (255,255,255) , coord(bomberPlane[t]) , 4)
            
        else:
            textRect1.center = coord(bomberPlane[t])
            textRect2.center = coord(fighterPlane[t])

            screen.blit(position_boomber , textRect1)
            screen.blit(position_fighter , textRect2)


        dist = calculateDistance(bomberPlane[t],fighterPlane[t])
        
        if dist <= 10:
            textRect4.center = (width / 2 , height / 2)
            screen.blit(position_match , textRect4)
            print("YES, at time {} and at Boomber pos {} and Fighter pos {}".format(t,bomberPlane[t],
                    fighterPlane[t]))
            running = False

        else:
            XF = fighterPlane[t][0] + round(VF * ((bomberPlane[t][0] - fighterPlane[t][0]) / dist))
            YF = fighterPlane[t][1] + round(VF * ((bomberPlane[t][1] - fighterPlane[t][1]) / dist))

            fighterPlane.append((XF,YF))

        pygame.display.flip()
        t += 1
             
    pygame.time.delay(1000)
    if not running:
        pygame.quit()
        break

    pygame.quit()