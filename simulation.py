import numpy as np
import taichi as ti
ti.init(arch=ti.gpu)

AGENT_N = 1024
SIMULI_N = 30
GRID_SIZE = 512
ORIGIN = ti.Vector([GRID_SIZE/2, GRID_SIZE/2])
INIT_R = 50
SIMULI_R = 200
SENSE_ANGLE = 0.20 * np.pi
SENSE_DIST = 4.0
EVAPORATION = 0.95
MOVE_ANGLE = 0.1 * np.pi
MOVE_STEP = 2.0 


TrailGrid = ti.field(dtype=ti.f32, shape=[2, GRID_SIZE, GRID_SIZE])
MoveGrid = ti.field(dtype=ti.f32, shape=[GRID_SIZE, GRID_SIZE])
TruthGrid = ti.field(dtype=ti.i32, shape=[GRID_SIZE, GRID_SIZE])

TruthGridDisp = ti.Vector.field(3, dtype=ti.f32, shape=[GRID_SIZE,GRID_SIZE])

SimuliPosition = ti.Vector.field(2, dtype=ti.f32, shape=[SIMULI_N])

Position = ti.Vector.field(2, dtype=ti.f32, shape=[AGENT_N])
Heading = ti.field(dtype=ti.f32, shape=[AGENT_N])

@ti.kernel
def init():
  for p in ti.grouped(TrailGrid):
    TrailGrid[p] = 0.0
  
  for i, j in ti.ndrange(GRID_SIZE, GRID_SIZE):
    MoveGrid[i, j] = 0.0
  
  for i in SimuliPosition:
    SimuliPosition[i] = ti.Vector([2 * ti.random() - 1, 2 * ti.random() - 1]) * SIMULI_R + ORIGIN
    ipos = SimuliPosition[i].cast(int)
    TruthGrid[ipos] = 2

  for i in Position:
    Position[i] = ti.Vector([ti.random(), ti.random()])* GRID_SIZE
    # Position[i] = ti.Vector([2*ti.random()-1, 2*ti.random()-1]) * INIT_R + ORIGIN
    Heading[i] = ti.random() * np.pi * 2.0

    ipos = Position[i].cast(int)
    TruthGrid[ipos] = 3


@ti.func
def sense(phase, pos, ang):
  p = pos + ti.Vector([ti.cos(ang), ti.sin(ang)]) * SENSE_DIST
  return TrailGrid[phase, p.cast(int) % GRID_SIZE]
  
@ti.kernel
def step(phase: ti.i32):
  #move
  for i in Position:
    pos, ang = Position[i], Heading[i]
    l = sense(phase, pos, ang - SENSE_ANGLE)
    c = sense(phase, pos, ang)
    r = sense(phase, pos, ang + SENSE_ANGLE)
    if l < c < r:
      ang += MOVE_ANGLE
    elif l > c > r:
      ang -= MOVE_ANGLE
    elif c < l and c < r:
      ang += MOVE_ANGLE * (2 * (ti.random() < 0.5) - 1)
    # change Truth
    TruthGrid[pos.cast(int)] = 0
    pos += ti.Vector([ti.cos(ang), ti.sin(ang)]) * MOVE_STEP
    Position[i], Heading[i] = pos, ang
    TruthGrid[pos.cast(int)] = 3


  # deposit
  for i in Position:
    ipos = Position[i].cast(int) % GRID_SIZE
    TrailGrid[phase, ipos] += 1.0
    # disp
    MoveGrid[ipos] += 1.0

  # simuli
  for i in SimuliPosition:
    simIPos = SimuliPosition[i].cast(int)
    TrailGrid[phase, simIPos] += 1.0
    TruthGrid[simIPos] = 2

  # diffuse
  for i, j in ti.ndrange(GRID_SIZE, GRID_SIZE):
    a = 0.0
    for di in ti.static(range(-1, 2)):
      for dj in ti.static(range(-1, 2)):
        a += TrailGrid[phase, (i + di) % GRID_SIZE, (j + dj) % GRID_SIZE]
        
    a *= EVAPORATION / 9.0
    TrailGrid[1 - phase, i, j] = a 
    
    if TruthGrid[i, j] == 0:
      TruthGridDisp[i, j] = ti.Vector([0.0, 0.0, 0.0])
    elif TruthGrid[i, j] == 1:
      TruthGridDisp[i, j] = ti.Vector([1.0, 1.0, 1.0])
    elif TruthGrid[i, j] == 2:
      TruthGridDisp[i, j] = ti.Vector([1.0, 1.0, 0.8])
    elif TruthGrid[i, j] == 3:
      TruthGridDisp[i, j] = ti.Vector([0.8, 0.5, 0.5])


print("[Hint] Press A/Z to change the simulation speed.")
gui = ti.GUI('Physarum')
init()
i = 0
step_per_frame = gui.slider('step_per_frame', 1, 100, 1)
while gui.running and not gui.get_event(gui.ESCAPE):
    for _ in range(int(step_per_frame.value)):
        step(i % 2)
            # Desp

        if (i % 50 == 0):
          filename = "./out/1024_Simuli_random/trail_{}.png".format(i)
          ti.imwrite(TrailGrid.to_numpy()[0], filename)
          filename = "./out/1024_Simuli_random/move_{}.png".format(i)
          ti.imwrite(MoveGrid.to_numpy(), filename)
          filename = "./out/1024_Simuli_random/truth_{}.png".format(i)
          ti.imwrite(TruthGridDisp.to_numpy(), filename)
        i += 1
    gui.set_image(TruthGridDisp.to_numpy())
    gui.show()