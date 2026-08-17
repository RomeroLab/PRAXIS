# Automated fridge

Hardware for robotic control of the laboratory refrigerator door in the PRAXIS
[`environment/`](../..). A 12 V linear actuator opens and closes a mini-fridge door under
control of an Arduino, allowing the [`lab_controller.py`](../../lab_controller.py) workflow to move
plates and reagents in and out of cold storage without human intervention.

## How it works

The Arduino reads two 5 V digital signals from the liquid handler and drives a 12 V linear actuator
through a pair of relays:

- **Signal 1 HIGH, Signal 2 LOW** → first relay activates → actuator **extends** (door opens).
- **Signal 1 LOW, Signal 2 HIGH** → second relay activates, reversing actuator polarity → actuator
  **retracts** (door closes).
- **Both LOW** → actuator stops.
- **Both HIGH** → treated as an error; all relays are switched off (safety stop).

A 50 ms delay separates relay switching so the two relays are never closed at once. The relays are
required because the actuator runs on 12 V while the Arduino outputs only 5 V: the 5 V Arduino
signal switches the relays, which gate 12 V supply power to the actuator.

## Contents

| File | Description |
|------|-------------|
| `actuator_control.cpp` | Arduino firmware (pin assignments, relay logic, safety stop) |
| `fridge_circuit_diagram.png` | Wiring schematic for the Arduino, relays, and actuator |
| `3dprinting_files/minifridge_mounting_post.stl` | Printable post that mounts the actuator to the fridge |
| `3dprinting_files/plate_offset_final.stl` | Printable plate-offset part |

## Build notes

Flash `actuator_control.cpp` to the Arduino and wire it per `fridge_circuit_diagram.png`. Pin
assignments in the firmware: relay control on pins 8 and 7, liquid-handler input signals on pins 10
and 11. The relays are active-low.
