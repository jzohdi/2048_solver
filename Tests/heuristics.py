def test_position_score(Heuristics):
    sample_board = [
        [64, 32, 16, 8],
        [32, 16, 8, 4],
        [4, 16, 32, 4],
        [2, 128, 0, 0]
    ]
    max_tile = 128
    score = Heuristics.position_score(sample_board, max_tile)
    print(score)


def test_distances(Heuristics):
    correct_pos_board = [[16, 15, 14, 13], [
        12, 11, 10, 9], [8, 7, 6, 5], [4, 3, 2, 1]]

    also_correct = [[128, 128, 128, 128], [128, 128, 128, 128],
                    [128, 128, 128, 128], [128, 128, 128, 128]]

    get_position_diff = Heuristics.desired_pos_approx_distance(
        correct_pos_board)
    print(f"diff score: {get_position_diff}")
    get_position_diff = Heuristics.desired_pos_approx_distance(also_correct)
    print(f"diff score: {get_position_diff}")

    average_case = [[256, 128, 64, 32], [
        64, 32, 16, 8], [32, 16, 8, 4], [4, 2, 0, 0]]
    get_position_diff = Heuristics.desired_pos_approx_distance(average_case)
    print(f"diff score: {get_position_diff}")

    very_bad = [average_case[3][::-1], average_case[2][::-1],
                average_case[1][::-1], average_case[0][::-1]]
    get_position_diff = Heuristics.desired_pos_approx_distance(very_bad)
    print(f"diff score: {get_position_diff}")


def test_find_top_vals(Heuristics):
    board = [[256, 128, 64, 32], [
        64, 32, 16, 8], [32, 16, 8, 4], [4, 2, 0, 0]]
    top_vals = Heuristics.find_top_vals(board, 16)
    print(top_vals[::-1])


def run_tests():
    import os
    import sys
    sys.path.append(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    from PlayerAI_3 import Heuristics, PlayerAI
    # player = PlayerAI()
    test_position_score(Heuristics)
    test_distances(Heuristics)
    test_find_top_vals(Heuristics)


if __name__ == "__main__":
    run_tests()
