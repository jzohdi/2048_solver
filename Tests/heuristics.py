def run_tests():
    import os
    import sys
    sys.path.append(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    from PlayerAI_3 import Heuristics, PlayerAI
    player = PlayerAI()
    sample_board = [
        [64, 32, 16, 8],
        [32, 16, 8, 4],
        [4, 16, 32, 4],
        [2, 128, 0, 0]
    ]
    max_tile = 128
    score = Heuristics.position_score(sample_board, max_tile)
    print(score)


if __name__ == "__main__":
    run_tests()
