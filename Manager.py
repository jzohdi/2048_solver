from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from PlayerAI_3 import PlayerAI, Heuristics
import os
import time


class Board:
    def __init__(self):
        self.max_tries = 4
        self.get_tiles_javascript = """
                const getTile = arr => {
                    const tile = [];
                    arr.forEach(str => {
                    let matched = str.match(/tile-\d/)
                    if(matched) {
                        tile[1] = str.match(/\d+/)[0]
                    }
                    matched = str.match(/tile-position-/);
                    if (matched) {
                        tile[0] = str.match(/[\d+]-[\d+]/)[0];
                    }
                    })
                    return tile
                }
                const getAll = () => {
                    let tiles = []
                    const x = document.getElementsByClassName("tile-container");
                    x[0].childNodes.forEach( ele => {
                        const curr = getTile(ele.classList)
                        tiles.push(curr)
                    })
                    return tiles
                }
                return getAll()
            """
        self.is_game_over_js = """
            const isGameOver = () => {
                const m = document.getElementsByClassName("game-message");
                return m[0].classList.contains("game-over")
                }
            return isGameOver()
            """
        self.retry_js = """
            document.getElementsByClassName("retry-button")[0].click();
        """
        self.get_score_js = """
            const score = () => {
                return document.getElementsByClassName("score-container")[0].innerText
            }
            return score()
        """
        self.actionDic = {
            0: Keys.UP,
            1: Keys.DOWN,
            2: Keys.LEFT,
            3: Keys.RIGHT
        }
        self.number_offspring = 5

    def get_key_from_move(self, move):

        direction = self.actionDic[move]
        return direction

    def is_game_over(self, browser):
        result = browser.execute_script(self.is_game_over_js)
        return result

    def get_score(self, browser):
        result = browser.execute_script(self.get_score_js)
        return int(result.split("\n")[0])

    def try_again(self, browser):
        browser.execute_script(self.retry_js)

    def parse_board(self, browser):
        try:
            els = browser.execute_script(self.get_tiles_javascript)
            self.filter_board(els)
        except:
            time.sleep(1)
            print("trying again...")
            return self.parse_board(browser)

    def filter_board(self, arr):

        board_dictionary = {}

        for tile in arr:
            pos = tile[0]
            if pos in board_dictionary:
                self.update_val(board_dictionary, pos, tile[1])
            else:
                board_dictionary[pos] = int(tile[1])

        self.set_board(board_dictionary)

    def set_board(self, board_dict):
        self.map = [[0]*4 for i in range(4)]

        for key, value in board_dict.items():
            key = key.split("-")
            row = int(key[0]) - 1
            col = int(key[1]) - 1
            self.map[col][row] = value

    def update_val(self, board_dictionary, pos, value):
        if board_dictionary[pos] < int(value):
            board_dictionary[pos] = int(value)

    def init_offspring(self, player, children, weights):
        offspring_of_best_weights = player.get_offspring(weights)
        for child in offspring_of_best_weights:
            children.append(child)


def getMaxTile(board):
    maxTile = 0

    for x in range(4):
        for y in range(4):
            maxTile = max(maxTile, board[x][y])

    return maxTile


def board_array_to_s(board):
    string = ''
    for row in board:
        for cell in row:
            string += str(cell) + "  "
        string += "\n"

    return string


def run_child(manager, player, browser, body):

    while True:
        manager.parse_board(browser)

        move_int = player.getMove(manager)
        # print(f'board:\n\n{board_array_to_s(manager.map)}')
        # score = Heuristics.rateBoard(
        #     player.weights_tuple(),
        #     manager.map,
        #     getMaxTile(manager.map),
        #     move_int,
        #     True
        # )
        # print(f"total:  {score}")
        if move_int != None:
            key = manager.get_key_from_move(move_int)
            body.send_keys(key)

        elif manager.is_game_over(browser):
            score = manager.get_score(browser)
            return (score, player.weights_tuple())


def play_2028(player, manager, browser, body):
    max_score = 0
    best_weights = player.weights_tuple()
    children_to_try = [best_weights]

    while True:

        try_weights = children_to_try.pop()
        player.set_weights(try_weights)
        # get score for single offpsring

        (score, child_weights) = run_child(manager, player, browser, body)
        print(f"score... {score}")
        if score > max_score:
            max_score = score
            best_weights = child_weights
            print(f"new best weights {best_weights}")
            with open("previous_best_weight.txt", "w+") as f:
                f.write(str(best_weights))
                f.close()
        if not children_to_try:
            manager.init_offspring(player, children_to_try, best_weights)

        manager.try_again(browser)


if __name__ == "__main__":

    browser = None

    try:
        player = PlayerAI()
        path = os.path.dirname(os.path.abspath(__file__))
        browser = webdriver.Chrome(ChromeDriverManager().install())
        browser.get(os.path.join(path, "index.html"))
        manager = Board()

        body = browser.find_element_by_tag_name("body")

        play_2028(player, manager, browser, body)
    except KeyboardInterrupt:

        if browser != None:
            browser.quit()

    except Exception as e:
        print("An error occured: ")
        print(e)

        if browser != None:
            browser.quit()
