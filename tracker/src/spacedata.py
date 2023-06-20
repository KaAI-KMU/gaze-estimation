# import keyboard

# i=0

# while True:
#     i += 1

#     keycode = keyboard.read_key()
#     if keycode == 'space':
#         print(i)
#         break
#     else:
#         if keycode:
#             print(keycode)
#             keycode=False

from pynput import keyboard
import pandas as pd

i=0

def on_press(key):
    print('Key %s pressed' % key)

def on_release(key):
    global i
    print('Key %s released' %key)
    if key == keyboard.Key.space:
        data = {'a':[i],'b':[i+1],'c':[i+2]}
        df=pd.DataFrame(data)
        df.to_csv('/home/kaai/catkin_ws/src/tracker/src/test/test.csv', mode='a', index=False, header=False)
        return False
    if key == keyboard.Key.esc: #esc 키가 입력되면 종료
        return False

while True:
    i += 1
    with keyboard.Listener(on_press=on_press,on_release=on_release) as listener:
        listener.join()
