import subprocess
for burst in range(20):
    children=[subprocess.Popen(['/bin/sleep','0.4']) for _ in range(32)]
    for child in children: child.wait()
print('640 short-lived children closed')
