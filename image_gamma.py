import cv2
import numpy as np
import time
import multiprocessing

def process_zone(zone, color, color1, bright, zones, size1, size2):
    part = np.zeros((len(color), 3))
    a = [[0, 0, 0]]
    n = 0
    for i in range(len(color)):
        a += color1[i] * chi(color[i], zone, zones)
        n += chi(color[i], zone, zones)
    cur_col = a / n
    for i in range(len(color)):
        part[i] = cur_col * chi(color[i], zone, zones)
    part = part.reshape(size1*size2, 3)
    return part

def chi(x, zone, zones):
    if np.allclose(x, zones[zone]):
        return 1
    else:
        return 0

if __name__ == '__main__':
    start = time.time()

    size1, size2 = 100, 100

    img = cv2.imread('med.png')
    img = img.reshape(-1, img.shape[-1])

    img1 = cv2.imread('med2.png')
    img1 = img1.reshape(-1, img1.shape[-1])

    color = np.empty((0, 3))
    bright = []
    color1 = np.empty((0, 3))
    bright1 = []
    for i in range(len(img)):
        current_bright = np.linalg.norm(img[i])
        current_bright1 = np.linalg.norm(img1[i])
        bright.append(current_bright)
        bright1.append(current_bright1)
        r, g, b = 0, 0, 0
        r1, g1, b1 = 0, 0, 0
        if current_bright != 0:
            r = img[i][0]/current_bright
            g = img[i][1]/current_bright
            b = img[i][2]/current_bright
        current_color = np.array([r, g, b])
        color = np.append(color, [current_color], axis = 0)
        if current_bright1 != 0:
            r1 = img1[i][0]/current_bright1
            g1 = img1[i][1]/current_bright1
            b1 = img1[i][2]/current_bright1
        current_color1 = np.array([r1, g1, b1])
        color1 = np.append(color1, [current_color1], axis = 0)

    print('complete transformation')

    unique_rows = [color[0]]
    for row in color:
        if not any(np.allclose(row, unique_row) for unique_row in unique_rows):
            unique_rows.append(row)

    zones = np.array(unique_rows)
    print(len(zones))

    num_processes = 6
    pool = multiprocessing.Pool(processes=num_processes)

    results = []
    for zone in range(len(zones)):
        results.append(pool.apply_async(process_zone, args=(zone, color, color1, bright, zones, size1, size2)))

    pvq = np.zeros((size1*size2, 3))
    counter = 1

    for r in results:
        print(counter)
        pvq += r.get()
        counter += 1

    pool.close()
    pool.join()

    res = np.zeros((size1*size2, 3))
    for i in range(len(res)):
        if bright[i] != 0.:
            res[i] = (color1[i] - pvq[i]) * bright[i]
            pvq[i] *= bright[i]
        else:
            res[i] = (color1[i] - pvq[i]) * (255 - bright[i])
            pvq[i] *= (255 - bright[i])       

    res = res.reshape(size1, size2, 3)
    pvq = pvq.reshape(size1, size2, 3)
    
    cv2.imwrite('pvq.png', pvq)
    cv2.imwrite('result.png', res)

    end = time.time()
    print(end - start)
