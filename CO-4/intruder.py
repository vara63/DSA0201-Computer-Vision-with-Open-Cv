import sys
from collections import deque


def main():
    data = sys.stdin.buffer.read().split()
    R, C, T, A = int(data[0]), int(data[1]), int(data[2]), int(data[3])
    n = R * C
    pos = 4
    bg = data[pos:pos + n]; pos += n          # background image B (row-major)
    fr = data[pos:pos + n]                    # current frame F
    # foreground mask: |F - B| > T  (strictly greater)
    fg = bytearray(n)
    for i in range(n):
        if abs(int(fr[i]) - int(bg[i])) > T:
            fg[i] = 1

    # (1) row-scan rule: count foreground pixels whose upper AND left neighbours are not foreground
    scan = 0
    for r in range(R):
        base = r * C
        for c in range(C):
            if fg[base + c]:
                up = r > 0 and fg[base - C + c]
                left = c > 0 and fg[base + c - 1]
                if not up and not left:
                    scan += 1

    # (2) 4-connected labelling with an explicit queue; area and bounding box collected on the way
    seen = bytearray(n)
    objects = []                              # (r1, c1, r2, c2, area) in order of first pixel (row-major)
    for start in range(n):
        if fg[start] and not seen[start]:
            seen[start] = 1
            q = deque([start])
            area = 0
            r1 = c1 = 10 ** 9
            r2 = c2 = -1
            while q:
                p = q.popleft()
                r, c = divmod(p, C)
                area += 1
                if r < r1: r1 = r
                if r > r2: r2 = r
                if c < c1: c1 = c
                if c > c2: c2 = c
                if r > 0 and fg[p - C] and not seen[p - C]:
                    seen[p - C] = 1; q.append(p - C)
                if r < R - 1 and fg[p + C] and not seen[p + C]:
                    seen[p + C] = 1; q.append(p + C)
                if c > 0 and fg[p - 1] and not seen[p - 1]:
                    seen[p - 1] = 1; q.append(p - 1)
                if c < C - 1 and fg[p + 1] and not seen[p + 1]:
                    seen[p + 1] = 1; q.append(p + 1)
            objects.append((r1, c1, r2, c2, area))

    # (3) discard noise (area < A); the list is already in row-major order of each object's first pixel
    kept = [o for o in objects if o[4] >= A]
    largest = max((o[4] for o in kept), default=0)
    boxes = ' '.join(f'{o[0]},{o[1]},{o[2]},{o[3]}' for o in kept) if kept else 'NONE'
    sys.stdout.write(f'Scan Count = {scan}\nObjects = {len(kept)}\nLargest Area = {largest}\nBoxes = {boxes}\n')


if __name__ == '__main__':
    main()
