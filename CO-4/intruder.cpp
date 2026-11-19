#include <bits/stdc++.h>
using namespace std;

int main() {
    int R, C, T, A;
    if (scanf("%d %d %d %d", &R, &C, &T, &A) != 4) return 0;
    int n = R * C;
    vector<int> B(n), F(n);
    for (int i = 0; i < n; i++) scanf("%d", &B[i]);
    for (int i = 0; i < n; i++) scanf("%d", &F[i]);
    vector<char> fg(n), seen(n, 0);
    for (int i = 0; i < n; i++) fg[i] = abs(F[i] - B[i]) > T;      // strictly greater than T

    long long scan = 0;                                          // row-scan rule
    for (int r = 0; r < R; r++)
        for (int c = 0; c < C; c++)
            if (fg[r * C + c]) {
                bool up = r > 0 && fg[(r - 1) * C + c], left = c > 0 && fg[r * C + c - 1];
                if (!up && !left) scan++;
            }

    struct Obj { int r1, c1, r2, c2, area; };
    vector<Obj> objs;
    vector<int> q(n);                                            // explicit queue (no recursion)
    const int dr[4] = {-1, 1, 0, 0}, dc[4] = {0, 0, -1, 1};
    for (int s = 0; s < n; s++) {
        if (!fg[s] || seen[s]) continue;
        int head = 0, tail = 0; q[tail++] = s; seen[s] = 1;
        Obj o{INT_MAX, INT_MAX, -1, -1, 0};
        while (head < tail) {
            int p = q[head++], r = p / C, c = p % C;
            o.area++; o.r1 = min(o.r1, r); o.r2 = max(o.r2, r); o.c1 = min(o.c1, c); o.c2 = max(o.c2, c);
            for (int k = 0; k < 4; k++) {
                int nr = r + dr[k], nc = c + dc[k];
                if (nr < 0 || nr >= R || nc < 0 || nc >= C) continue;
                int np = nr * C + nc;
                if (fg[np] && !seen[np]) { seen[np] = 1; q[tail++] = np; }
            }
        }
        objs.push_back(o);                                       // discovery order = row-major order of first pixel
    }
    vector<Obj> kept;
    for (auto &o : objs) if (o.area >= A) kept.push_back(o);
    int largest = 0;
    for (auto &o : kept) largest = max(largest, o.area);
    printf("Scan Count = %lld\nObjects = %d\nLargest Area = %d\nBoxes =", scan, (int)kept.size(), largest);
    if (kept.empty()) printf(" NONE");
    for (size_t i = 0; i < kept.size(); i++) printf("%s%d,%d,%d,%d", i ? " " : " ", kept[i].r1, kept[i].c1, kept[i].r2, kept[i].c2);
    printf("\n");
    return 0;
}
