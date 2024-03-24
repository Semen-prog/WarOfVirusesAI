#include <iostream>
#include <fstream>
#include <string>

using namespace std;

const int MAXN = 303;
const int MAXM = MAXN;
const int BAD = -100;

int n, m, k, timer;
int A[MAXN][MAXM], used[MAXN][MAXM];
ofstream lout("log.txt", ofstream::out);

void out(int pl) {
	cout << pl << endl;
	lout << -1 << ' ' << -1 << ' ' << -1 << endl;
}

inline int get_cell(int x, int y) {
	if (x < 0 || n < x || y < 0 || m < y)
		return BAD;
	return A[x][y];
}

bool dfs(int pl, int x, int y) {
	used[x][y] = timer;
	for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            int c = get_cell(x + dx, y + dy);
            if (c == pl) return true;
            if (c == -pl && (used[x + dx][y + dy] != timer || dfs(pl, x + dx, y + dy)))
                return true;
        }
    }
	return false;
}

bool make_move(int pl, int x, int y, string &output) {
	int c = get_cell(x, y);
	if (c < 0 || c == pl) return false;
	++timer;
	if (!dfs(pl, x, y)) return false;
	A[x][y] = (!c ? pl : -pl);
	output += to_string(x);
	output += ' ';
	output += to_string(y);
	output += ' ';
	output += to_string(A[x][y]);
	output += '\n';
	return true;
}

void dfs2(int pl, int x, int y) {
	used[x][y] = timer;
	for (int dx = -1; dx <= 1; ++dx)
		for (int dy = -1; dy <= 1; ++dy) {
			int c = get_cell(x + dx, y + dy);
			if (c != -pl && c < 0) continue;
			if (used[x + dx][y + dy] != timer)
				dfs2(pl, x + dx, y + dy);
		}
}

int hard(int pl) {
	++timer;
	for (int x = 0; x < n; ++x)
		for (int y = 0; y < m; ++y)
			if (A[x][y] == pl && used[x][y] != timer)
				dfs2(pl, x, y);
	int scores[k + 1]{0};
	for (int x = 0; x < n; ++x)
		for (int y = 0; y < m; ++y) {
			if (used[x][y] == timer) scores[pl]++;
			else if (A[x][y] != BAD) scores[abs(A[x][y])]++;
		}
//	cerr << pl_score << ',' << op_score << endl;
	int best = -1;
    for (int i = 1; i <= k; ++i) {
        if (best == -1 || scores[i] > scores[best]) {
            best = i;
        }
    }
    return best;
}

int main(int argc, char* argv[]) {
	cin >> n >> m >> k;
	lout << n << ' ' << m << ' ' << k << endl;
	for (int x = 0; x < n; ++x) {
		for (int y = 0; y < m; ++y) {
			cin >> A[x][y];
			lout << A[x][y] << ' ';
		}
		lout << endl;
	}
    int skipped_cnt = 0;
	while (true) {
		int pl, a;
        cin >> pl >> a;
        bool bad_move = false;
        if (skipped_cnt == k - 1) {
            out(hard(pl));
            return 0;
        }
        if (a == 0) {
            skipped_cnt++;
            bad_move = true;
        } else {
            skipped_cnt = 0;
        }
        string output;
        pair<pair<int, int>, int> rollback[3];
		for (int i = 0; i < a; ++i) {
			int x, y; cin >> x >> y, --x, --y;
            if (!bad_move) {
                rollback[i] = {{x, y}, A[x][y]};
                if (!make_move(pl, x, y, output)) {
                    bad_move = true;
                    for (int j = 0; j < i; ++j) {
                        A[rollback[j].first.first][rollback[j].first.second] = rollback[j].second;
                    }
                }
            }
		}
        if (!bad_move) {
            cout << 0 << endl;
            lout << output;
        } else {
            cout << -1 << endl;
        }
		if (argc >= 2) {
			for (int x = 0; x < n; ++x) {
				for (int y = 0; y < m; ++y)
					cerr << A[x][y] << ' ';
				cerr << endl;
			}
			cerr << endl;
		}
	}
}