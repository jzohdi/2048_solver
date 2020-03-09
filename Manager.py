from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from PlayerAI_3 import PlayerAI
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
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT"
        }
        self.number_offspring = 10

    def get_key_from_move(self, move):

        direction = self.actionDic[move]

        if direction == "UP":
            return Keys.UP
        if direction == "DOWN":
            return Keys.DOWN
        if direction == "LEFT":
            return Keys.LEFT
        if direction == "RIGHT":
            return Keys.RIGHT

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
        for i in range(self.number_offspring):
            children.append(player.get_offspring(weights))


if __name__ == "__main__":

    player = PlayerAI()
    path = os.path.dirname(os.path.abspath(__file__))
    browser = webdriver.Chrome(ChromeDriverManager().install())
    browser.get(os.path.join(path, "index.html"))
    manager = Board()

    body = browser.find_element_by_tag_name("body")
    # game_over = browser.find_element_by_class_name("retry-button")
    # print(game_over)
    max_score = 0
    best_weights = player.weights_tuple()
    children_to_try = [best_weights]

    while True:

        try_weights = children_to_try.pop()
        player.set_weights(try_weights)
        # get score for single offpsring
        while True:

            manager.parse_board(browser)
            move_int = player.getMove(manager)
            # print(move_int)
            if move_int != None:
                key = manager.get_key_from_move(move_int)
                body.send_keys(key)

            elif manager.is_game_over(browser):
                score = manager.get_score(browser)
                print(f"score... {score}")
                if score > max_score:
                    max_score = score
                    best_weights = player.weights_tuple()
                    print(f"new best weights {best_weights}")

                break

        if not children_to_try:
            manager.init_offspring(player, children_to_try, best_weights)

        manager.try_again(browser)
        # browser.quit()
