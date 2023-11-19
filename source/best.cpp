#include <bits/stdc++.h>
#define all(a) a.begin(), a.end()

using namespace std;

int n, m;
vector<vector<int>> field;

const vector<pair<int, int>> table_paths = { {0, 1}, {0, -1}, {1, 0}, {-1, 0}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1} };
const int MAX_TURN_CNT = 3, inf = 1'000'000'000;

bool is_correct(int i, int j) {
    return 0 <= i && i < n && 0 <= j && j < m;
}

bool is_our_cross(int i, int j) {
    return field[i][j] == 0;
}

bool is_our_tower(int i, int j) {
    return field[i][j] == 1;
}

bool is_opponent_cross(int i, int j) {
    return field[i][j] == 2;
}

bool is_opponent_tower(int i, int j) {
    return field[i][j] == 3;
}

bool is_empty_point(int i, int j) {
    return field[i][j] == 4;
}

bool is_mountain(int i, int j) {
    return field[i][j] == 5;
}

void calc_bfs(string mode, vector<vector<pair<int, pair<int, int>>>>& dist, vector<vector<int>>& is_available) {
    dist.assign(n, vector<pair<int, pair<int, int>>>(m, { inf, {-1, -1} }));
    deque<pair<int, int>> points_queue;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if ((is_available[i][j] && mode == "player's available") ||
                (is_our_tower(i, j) && mode == "player's tower") ||
                (is_available[i][j] && mode == "opponent's available") ||
                (is_opponent_tower(i, j) && mode == "opponent's tower")) 
                points_queue.emplace_back(i, j);
        }
    }
    for (auto [i, j]: points_queue) {
        dist[i][j] = { 0, {-1, -1} };
    }
    while (!points_queue.empty()) {
        auto [i, j] = points_queue.front();
        points_queue.pop_front();
        if (dist[i][j].first != 0 && !is_empty_point(i, j)) continue;
        int d = dist[i][j].first;
        for (auto [di, dj]: table_paths) {
            int i_new = i + di, j_new = j + dj;
            if (!is_correct(i_new, j_new) || dist[i_new][j_new].first != inf) continue;
            dist[i_new][j_new] = { d + 1, {i, j} };
            points_queue.emplace_back(i_new, j_new);
        }
    }
}

void bfs(vector<vector<int>>& dist) {
    dist.assign(n, vector<int>(m, inf));
    deque<pair<int, int>> points_queue;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if (is_empty_point(i, j)) 
                points_queue.emplace_back(i, j);
        }
    }
    for (auto [i, j]: points_queue) {
        dist[i][j] = 0;
    }
    while (!points_queue.empty()) {
        auto [i, j] = points_queue.front();
        points_queue.pop_front();
        int d = dist[i][j];
        for (auto [di, dj]: table_paths) {
            int i_new = i + di, j_new = j + dj;
            if (!is_correct(i_new, j_new) || dist[i_new][j_new] != inf || (!is_empty_point(i_new, j_new) && !is_opponent_cross(i_new, j_new))) continue;
            dist[i_new][j_new] = d + 1;
            points_queue.emplace_back(i_new, j_new);
        }
    }
}

void calc_dfs(int i, int j, vector<vector<int>>& used) {
    used[i][j] = 1;
    for (auto [di, dj]: table_paths) {
        int i_new = i + di, j_new = j + dj;
        if (!is_correct(i_new, j_new) || used[i_new][j_new] || (!is_our_tower(i_new, j_new) && !is_our_cross(i_new, j_new))) continue;
        calc_dfs(i_new, j_new, used);
    }
}

vector<pair<int, int>> make_turn() {
    vector<vector<pair<int, pair<int, int>>>> dist_opponent_available, dist_player_available, dist_opponent_tower, dist_player_tower;
    vector<vector<int>> player_available(n, vector<int>(m, 0)), opponent_available(n, vector<int>(m, 0)), in_point;
    vector<vector<int>> cannot_eat_connector(n, vector<int>(m, 0)), is_connector(n, vector<int>(m, 0));
    int turns = MAX_TURN_CNT;
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if (is_our_cross(i, j) && !player_available[i][j]) calc_dfs(i, j, player_available);
            if (is_opponent_cross(i, j) && !opponent_available[i][j]) calc_dfs(i, j, opponent_available);
        }
    }

    calc_bfs("player's available", dist_player_available, player_available); calc_bfs("player's tower", dist_player_tower, player_available);
    calc_bfs("opponent's available", dist_opponent_available, opponent_available), calc_bfs("opponent's tower", dist_opponent_tower, opponent_available);
    bfs(in_point);

    set<tuple<int, int, int, int, int, int>> to_eat;
    map<pair<int, int>, int> is_eaten;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if (!is_our_cross(i, j)) continue;
            is_connector[i][j] = dist_player_tower[i][j].first == 1;
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if (!is_opponent_cross(i, j)) continue;
            cannot_eat_connector[i][j] = 1;
            for (auto [di, dj]: table_paths) {
                int i_new = i + di, j_new = j + dj;
                if (is_correct(i_new, j_new) && is_connector[i_new][j_new]) {
                    cannot_eat_connector[i][j] = 0;
                }
            } 
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if (!is_opponent_cross(i, j) || dist_player_available[i][j].first >= MAX_TURN_CNT) continue;
            to_eat.emplace(dist_player_available[i][j].first, cannot_eat_connector[i][j], in_point[i][j], dist_opponent_tower[i][j].first, i, j);
            is_eaten[{ i, j }] = dist_player_available[i][j].first;
        }
    }

    vector<pair<int, int>> result;

    while (!to_eat.empty()) {
        auto [dist, __, _, ___, i, j] = *to_eat.begin();
        //cerr << "tdist = " << dist << '\n';
        if (turns < dist) break;
        turns -= dist;
        //cerr << "no turns = " << turns << '\n';
        to_eat.erase(to_eat.begin());

        if (dist == MAX_TURN_CNT - 1) {
            auto [pi, pj] = dist_player_available[i][j].second;
            result.emplace_back(pi, pj);
            field[pi][pj] = 0;
        }
        result.emplace_back(i, j);
        field[i][j] = 1;
        for (auto [di, dj]: table_paths) {
            int i_new = i + di, j_new = j + dj;
            if (!is_correct(i_new, j_new) || !is_opponent_cross(i_new, j_new)) continue;
            if (is_eaten.find({ i_new, j_new }) != is_eaten.end()) {
                to_eat.erase({ is_eaten[{ i_new, j_new }], cannot_eat_connector[i_new][j_new], in_point[i_new][j_new], dist_opponent_tower[i_new][j_new].first, i_new, j_new });
            }
            to_eat.emplace(1, cannot_eat_connector[i_new][j_new], in_point[i_new][j_new], dist_opponent_tower[i_new][j_new].first, i_new, j_new);
            //cerr << "was added " << i_new << ' ' << j_new << ' ' << field[i_new][j_new] << ' ' << is_opponent_cross(i_new, j_new) << '\n';
            is_eaten[{ i_new, j_new }] = 1;
        }
    }

    if (turns == 0) return result;

    to_eat.clear(); is_eaten.clear();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if (!player_available[i][j]) continue;
            for (auto [di, dj]: table_paths) {
                int i_new = i + di, j_new = j + dj;
                if (!is_correct(i_new, j_new) || !is_empty_point(i_new, j_new) || dist_opponent_available[i_new][j_new].first < MAX_TURN_CNT) continue;
                to_eat.emplace(dist_player_tower[i_new][j_new].first, (i_new + j_new) % 2, -dist_opponent_available[i_new][j_new].first, i_new, j_new, 0);
                is_eaten[{ i_new, j_new }] = 1;
            }
        }
    }

    while (!to_eat.empty() && turns--) {
        auto [_, __, dist, i, j, ___] = *to_eat.begin();
        to_eat.erase(to_eat.begin());

        result.emplace_back(i, j);
        field[i][j] = 0;
        for (auto [di, dj]: table_paths) {
            int i_new = i + di, j_new = j + dj;
            if (!is_correct(i_new, j_new) || !is_empty_point(i_new, j_new) || is_eaten.find({ i_new, j_new }) != is_eaten.end() || dist_opponent_available[i_new][j_new].first <= MAX_TURN_CNT) continue;
            to_eat.emplace(dist_player_tower[i_new][j_new].first, (i_new + j_new) % 2, -dist_opponent_available[i_new][j_new].first, i_new, j_new, 0);
            is_eaten[{ i_new, j_new }] = 1;
        }
    }

    if (turns == MAX_TURN_CNT) {
        tuple<int, int, int> opt_point = { -1, -1, -1 };
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (!player_available[i][j]) continue;
                for (auto [di, dj]: table_paths) {
                    int i_new = i + di, j_new = j + dj;
                    if (!is_correct(i_new, j_new) || !is_empty_point(i_new, j_new)) continue;
                    opt_point = max(opt_point, make_tuple(dist_opponent_available[i_new][j_new].first, i_new, j_new));
                }
            }
        }
        auto [d, i, j] = opt_point;
        if (d == -1) return {};
        field[i][j] = 0;
        return { {i, j} };
    }

    return result;
}

int main() {
    cin >> n >> m;
    field.assign(n, vector<int>(m, -1));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            cin >> field[i][j];
        }
    }
    auto result = make_turn();
    cout << result.size() << '\n';
    for (auto [i, j]: result) {
        cout << i + 1 << ' ' << j + 1 << '\n';
    }
}