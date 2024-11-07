
import cv2

import pyautogui
import time
import pynput

from move import *

COLOR_ERROR_MAX = 62025  # 255 * 255

mouse_down = False


def on_click(x, y, button, pressed):
    global mouse_down
    mouse_down = pressed


class Detector:
    def __init__(self):
        self.board_left_coordinate = 0
        self.board_top_coordinate = 0
        self.precise_square_size = 0.0
        self.square_size = 0
        self.square_margin_size = 0

        self.light_square = 0
        self.dark_square = 0
        self.empty_error = 10

        self.detection_noise_threshold = 3 * self.square_margin_size

        self.skip = 0

        self.piece_grayscales = []

        self.pieces_detected = False
        try:
            for piece_type_temp in PIECE_NAMES_FEN.keys():
                self.piece_grayscales.append(cv2.cvtColor(cv2.imread("./detected_pieces/" + piece_type_temp + ".png"),
                                                          cv2.COLOR_BGR2GRAY))
            self.pieces_detected = True

        except FileNotFoundError:
            print("No pieces detected")
            self.pieces_detected = False

    def get_board_coordinates(self):
        pyautogui_ss = pyautogui.screenshot()
        pyautogui_ss.save("board_detection_screenshot.png")

        screenshot = cv2.imread('board_detection_screenshot.png')
        screenshot_grayscale = cv2.cvtColor(np.array(pyautogui_ss), cv2.COLOR_BGR2GRAY)
        screenshot_blur = cv2.GaussianBlur(screenshot_grayscale, (5, 5), 0)

        edges = cv2.Canny(screenshot_blur, 50, 150, apertureSize=3)

        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        largest_area = 0
        largest_rectangle = None

        for contour in contours:
            # Approximate contour to a polygon
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)

            # Check for rectangular shape (chessboard likely to be a rectangle)
            if len(approx) == 4:
                area = cv2.contourArea(contour)

                pts = np.array([point[0] for point in contour])

                rect = np.zeros((4, 2), dtype="float32")
                s = pts.sum(axis=1)
                rect[0] = pts[np.argmin(s)]
                rect[2] = pts[np.argmax(s)]

                diff = np.diff(pts, axis=1)
                rect[1] = pts[np.argmin(diff)]
                rect[3] = pts[np.argmax(diff)]

                width = np.linalg.norm(rect[0] - rect[1])
                height = np.linalg.norm(rect[0] - rect[3])

                if area > largest_area and abs(width - height) <= 10:
                    largest_rectangle = contour
                    largest_area = area

                # Draw the detected contour
                cv2.drawContours(screenshot, [approx], -1, (0, 255, 0), 3)

        if largest_rectangle is not None:
            # perimeter = cv2.arcLength(largest_rectangle, True)
            # rect_approx = cv2.approxPolyDP(largest_rectangle, 0.01 * perimeter, True)

            # cv2.drawContours(screenshot, [rect_approx], -1, (255, 0, 255), 3)
            pts = np.array([point[0] for point in largest_rectangle])

            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]

            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]

            print("Chessboard Corners:")
            print("Top Left:", rect[0])
            print("Top Right:", rect[1])
            print("Bottom Right:", rect[2])
            print("Bottom Left:", rect[3])

            # Calculate width and height of the chessboard
            width = np.linalg.norm(rect[0] - rect[1])
            height = np.linalg.norm(rect[0] - rect[3])
            print("Width:", width, "Height:", height)

            self.precise_square_size = width / 8
            self.square_size = int(width // 8)
            self.square_margin_size = int(self.precise_square_size // 15)

            self.board_left_coordinate = int(rect[0][0])
            self.board_top_coordinate = int(rect[0][1])

            print(self.board_left_coordinate, self.board_top_coordinate, self.square_size, self.square_margin_size)

            self.light_square = int(screenshot_grayscale
                    [self.board_top_coordinate + self.square_size * 2 + self.square_margin_size]
                    [self.board_left_coordinate + self.square_margin_size])

            self.dark_square = int(screenshot_grayscale
                    [self.board_top_coordinate + self.square_size * 2 + self.square_margin_size]
                    [self.board_left_coordinate + self.square_size + self.square_margin_size])

            self.skip = (self.square_size - 2 * self.square_margin_size) // 20

            return True
            # cv2.imshow('Detected Chessboard', screenshot)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        '''
        ret, corners = cv2.findChessboardCorners(screenshot_grayscale, (7, 7), None)

        if ret:
            # -- Draw and display the corners --
            
            img = cv2.drawChessboardCorners(screenshot, (7, 7), corners, ret)
            cv2.imshow('Chessboard', img)
            cv2.waitKey(0)
            

            self.square_size = int(corners[1][0][0] - corners[0][0][0])
            self.square_margin_size = self.square_size // 15

            self.board_left_coordinate = int(corners[0][0][0] - self.square_size)
            self.board_top_coordinate = int(corners[0][0][1] - self.square_size)

            self.light_square = int(screenshot_grayscale[self.board_top_coordinate + self.square_margin_size] \
                                                        [self.board_left_coordinate + self.square_margin_size])

            self.dark_square = int(screenshot_grayscale \
                    [self.board_top_coordinate + self.square_margin_size] \
                    [self.board_left_coordinate + self.square_size + self.square_margin_size])

            # print(self.light_square, self.dark_square)
            # print(self.square_size, self.square_margin_size, self.board_left_coordinate, self.board_top_coordinate)
        '''

        return False

    def get_pieces(self):
        # board top left corner coords (change if needed)

        # take a screenshot and store it locally
        pyautogui_ss = pyautogui.screenshot()
        pyautogui_ss.save("piece_detection_screenshot.png")

        # load local screenshot
        screenshot = cv2.imread('piece_detection_screenshot.png')
        image_grayscale = cv2.cvtColor(np.array(pyautogui_ss), cv2.COLOR_BGR2GRAY)

        self.piece_grayscales = []

        # loop over board rows
        for piece_code_str, val in PIECE_NAMES_INDEX.items():
            piece_name = val[0]
            row, col = val[1]

            y = int(self.board_top_coordinate + row * self.precise_square_size)
            x = int(self.board_left_coordinate + col * self.precise_square_size)

            # print(piece_name, row, col, y, x)
            # print(screenshot_grayscale[int(y + self.square_margin_size)][int(x + self.square_margin_size)])

            # crop piece image
            piece_image = \
                screenshot[int(y + self.square_margin_size):
                           int(y + self.precise_square_size - self.square_margin_size),

                           int(x + self.square_margin_size):
                           int(x + self.precise_square_size - self.square_margin_size)]

            # uncomment to display extracted images
            '''
            cv2.imshow('scr', piece_image)
            cv2.waitKey(0)
            '''

            # store extracted image
            cv2.imwrite('./detected_pieces/' + piece_name + '.png', piece_image)

            self.piece_grayscales.append(image_grayscale[

                                         int(y + self.square_margin_size):
                                         int(y + self.precise_square_size - self.square_margin_size),

                                         int(x + self.square_margin_size):
                                         int(x + self.precise_square_size - self.square_margin_size)])

        # clean up windows
        cv2.destroyAllWindows()
        return True

    def get_piece_diffs(self, location, piece, image_grayscale):
        x = location[0]
        y = location[1]

        empty_diff = 0
        color_diff = 0

        for y_coord in range(0, int(self.square_size - 2 * self.square_margin_size), self.skip):
            for x_coord in range(0, int(self.square_size - 2 * self.square_margin_size), self.skip):

                grayscale_value = int(image_grayscale[int(y + y_coord + self.square_margin_size)]
                                      [int(x + x_coord + self.square_margin_size)])

                test_empty = False

                if abs(grayscale_value - self.light_square) <= self.empty_error or \
                        abs(grayscale_value - self.dark_square) <= self.empty_error:
                    test_empty = True

                if piece == EMPTY:
                    if not test_empty:
                        empty_diff += 1
                    continue

                piece_grayscale_value = int(self.piece_grayscales[piece][y_coord][x_coord])

                piece_empty = False

                if abs(piece_grayscale_value - self.light_square) <= self.empty_error or \
                        abs(piece_grayscale_value - self.dark_square) <= self.empty_error:
                    piece_empty = True

                # Either the pixel is not empty and the matching piece's pixel is, or
                # the matching piece's pixel is not empty and the square is empty
                if abs(test_empty - piece_empty) == 1:
                    empty_diff += 1
                    continue

                # No diff calculations if both are empty
                if test_empty and piece_empty:
                    continue

                current_diff = abs(grayscale_value - piece_grayscale_value)

                # If the difference in pixel grayscale is extremely close, don't count it towards diff
                if current_diff > self.empty_error:
                    color_diff += current_diff * current_diff

        return empty_diff, color_diff

    def get_new_basic_fen(self, main_state):

        # start_time = time.time()
        # print(self.skip)

        # take a board snapshot
        pyautogui_ss = pyautogui.screenshot()
        # pyautogui_ss.save("detection_new.png")

        # load local screenshot
        # screenshot = cv2.imread('detection_new.png')

        image_grayscale = cv2.cvtColor(np.array(pyautogui_ss), cv2.COLOR_BGR2GRAY)
        fen = ''

        pixel_area = (int(self.square_size - 2 * self.square_margin_size) // self.skip) ** 2

        prediction_c = 0
        prediction_t = 0

        # board top left corner coords

        for row in range(8):
            # empty square counter
            empty = 0

            # loop over board columns
            for col in range(8):

                is_empty = False

                p_row = abs(row - 7 * main_state.perspective)
                p_col = abs(col - 7 * main_state.perspective)

                y = int(self.board_top_coordinate  + p_row * self.precise_square_size)
                x = int(self.board_left_coordinate + p_col * self.precise_square_size)

                # Proper location for empty square detection, y is higher
                central_square_x = int(x + self.square_size / 2)
                central_square_y = int(y + self.square_size / 2 + 2 * self.square_margin_size)

                # Quickly test if the square is empty or not
                num_empty = 0
                for test_center_y in range(central_square_y - 2, central_square_y + 3):
                    for test_center_x in range(central_square_x - 2, central_square_x + 3):
                        grayscale_value = int(image_grayscale[test_center_y][test_center_x])
                        if abs(grayscale_value - self.light_square) <= self.empty_error:
                            num_empty += 1

                        if abs(grayscale_value - self.dark_square) <= self.empty_error:
                            num_empty += 1

                        # Skip looking at this square since it is empty
                        if num_empty >= 15:
                            is_empty = True
                            break

                    if is_empty:
                        break

                # Detect Piece with Nearest Neighbor
                if not is_empty:
                    # Loop over piece types
                    '''
                    cropped_ss = screenshot[y + self.square_margin_size:y + self.square_size - self.square_margin_size,
                                            x + self.square_margin_size:x + self.square_size - self.square_margin_size]

                    cv2.imshow('img', cropped_ss)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    '''

                    best_empty_diff = -1
                    best_piece = EMPTY

                    piece_color_diffs = [-1] * 12

                    predicted_piece = main_state.position.board[STANDARD_TO_MAILBOX[row * 8 + col]]

                    if predicted_piece != EMPTY:
                        predicted_piece_flipped = (predicted_piece + 6) % 12
                        pieces_to_check = [predicted_piece, predicted_piece_flipped]

                        for piece in range(12):
                            if piece != predicted_piece and piece != predicted_piece_flipped:
                                pieces_to_check.append(piece)
                    else:
                        pieces_to_check = [i for i in range(12)]

                    for piece in pieces_to_check:

                        empty_diff, color_diff = self.get_piece_diffs((x, y), piece, image_grayscale)

                        piece_color_diffs[piece] = color_diff
                        # print(piece, (row, col), color_diff, empty_diff)

                        # Calculate Nearest Neighbour based on empty pixel differences
                        if best_empty_diff == -1 or empty_diff < best_empty_diff:
                            best_empty_diff = empty_diff
                            best_piece = piece

                            if best_empty_diff <= pixel_area / 40:
                                break

                    # Based on the best empty pixel differences, we can narrow the correct piece down to
                    # 2 options, which are the two colored pieces of the best piece type. The correct one is chosen
                    # based on the color differences
                    best_piece_flipped = (best_piece + 6) % 12
                    if piece_color_diffs[best_piece_flipped] == -1:
                        piece_color_diffs[best_piece_flipped] = \
                            self.get_piece_diffs((x, y), best_piece_flipped, image_grayscale)[1]

                    # print((p_row, p_col), best_piece, piece_color_diffs[best_piece],
                    #                                   piece_color_diffs[best_piece_flipped])

                    best_piece = best_piece if piece_color_diffs[best_piece] < \
                                               piece_color_diffs[best_piece_flipped] else best_piece_flipped

                    prediction_t += 1
                    if best_piece == predicted_piece:
                        prediction_c += 1

                    if empty:
                        fen += str(empty)
                        empty = 0

                    fen += list(PIECE_NAMES_FEN.values())[best_piece]

                else:
                    empty += 1

            if empty:
                fen += str(empty)
            if row < 7:
                fen += '/'

        # print(time.time() - start_time)

        print(fen, prediction_c / prediction_t)
        return fen

    def detect_move_change(self, main_state):

        pixel_area = (int(self.square_size - 2 * self.square_margin_size) // self.skip) ** 2

        pyautogui_ss = pyautogui.screenshot()
        image_grayscale = cv2.cvtColor(np.array(pyautogui_ss), cv2.COLOR_BGR2GRAY)

        legal_moves = main_state.position.get_legal_moves()

        origin_squares = set({})
        for move in legal_moves:
            origin_squares.add(get_from_square(move))

        origin_square = -1

        best_num_empty = 0
        possible_origins = []

        for origin_square_test in origin_squares:
            # Quickly test if the square is empty or not

            origin_square_standard = MAILBOX_TO_STANDARD[origin_square_test]

            row = origin_square_standard // 8
            col = origin_square_standard % 8

            p_row = abs(row - 7 * main_state.perspective)
            p_col = abs(col - 7 * main_state.perspective)

            y = int(self.board_top_coordinate  + p_row * self.precise_square_size)
            x = int(self.board_left_coordinate + p_col * self.precise_square_size)

            central_square_x = int(x + self.square_size / 2)
            central_square_y = int(y + self.square_size / 2 + 2 * self.square_margin_size)

            num_empty = 0

            # The amount of pixels to check on each side from the center,
            # forming a (2*4 / pixels_skip)^2 amount of pixels checked
            pixels_range = 4
            pixels_skip = 2
            pixels_square = (2 * pixels_range / pixels_skip) ** 2

            is_empty = False

            for test_center_y in range(central_square_y - (pixels_range - 1),
                                       central_square_y + pixels_range, pixels_skip):

                for test_center_x in range(central_square_x - (pixels_range - 1),
                                           central_square_x + pixels_range, pixels_skip):

                    grayscale_value = int(image_grayscale[test_center_y][test_center_x])
                    if abs(grayscale_value - self.light_square) <= self.empty_error:
                        num_empty += 1

                    if abs(grayscale_value - self.dark_square) <= self.empty_error:
                        num_empty += 1

            # Skip looking at this square since it is empty
            if num_empty >= pixels_square / 3 * 2:
                is_empty = True
                possible_origins.append(origin_square_test)

            if is_empty and num_empty > best_num_empty:
                # print(num_empty, pixels_square)
                best_num_empty = num_empty
                origin_square = origin_square_test

        # No possible origin squares which are now empty
        if origin_square == -1:
            return NO_MOVE

        # print(MAILBOX_TO_STANDARD[origin_square])

        if len(possible_origins) > 2:
            print("BIG CHANGES, POSITION LIKELY FLAWED")
            return NO_MOVE

        # Castle Handling
        # If two squares where pieces can move from are now empty, the move must have been castling, and the move
        # must have been made since both the rook and king moved.
        if len(possible_origins) == 2:
            # two castling moves could be possible, queen or king side, so the correct one corresponding to the rook
            # must be found
            castling_moves = []
            rook_square = 0

            for origin_square_test in possible_origins:
                for move in legal_moves:
                    if get_from_square(move) == origin_square_test:
                        if get_move_type(move) == MOVE_TYPE_CASTLE:
                            castling_moves.append(move)
                        elif get_selected(move) == WHITE_ROOK or get_selected(move) == BLACK_ROOK:
                            rook_square = get_from_square(move)

            if len(castling_moves) == 0 or rook_square == 0:
                print("BIG CHANGES, POSITION LIKELY FLAWED")
                return NO_MOVE

            if len(castling_moves) == 1:
                return castling_moves[0]
            else:
                assert(len(castling_moves) == 2)
                return castling_moves[0] if abs(get_to_square(castling_moves[0]) - rook_square) < \
                                            abs(get_to_square(castling_moves[1]) - rook_square) else \
                       castling_moves[1]

        lowest_diff_move = NO_MOVE
        lowest_diff_move_score = -1

        for move in legal_moves:

            if get_from_square(move) == origin_square:
                target_square = get_to_square(move)
                target_square_standard = MAILBOX_TO_STANDARD[target_square]
                occupied = get_occupied(move)
                selected = get_selected(move)

                row = target_square_standard // 8
                col = target_square_standard % 8

                p_row = abs(row - 7 * main_state.perspective)
                p_col = abs(col - 7 * main_state.perspective)

                y = int(self.board_top_coordinate + p_row * self.precise_square_size)
                x = int(self.board_left_coordinate + p_col * self.precise_square_size)

                best_piece = None
                best_diff = -1

                piece_checks = [occupied]

                if get_move_type(move) == MOVE_TYPE_PROMOTION:
                    piece_checks.append(get_promotion_piece(move))
                else:
                    piece_checks.append(selected)

                piece_color_diffs = [-1] * 12

                for piece in piece_checks:

                    empty_diff, color_diff = self.get_piece_diffs((x, y), piece, image_grayscale)

                    if piece != EMPTY:
                        piece_color_diffs[piece] = color_diff

                    if best_diff == -1 or empty_diff < best_diff:
                        best_diff = empty_diff
                        best_piece = piece

                        if best_diff <= pixel_area / 40:
                            break

                best_piece_flipped = (best_piece + 6) % 12

                # print(get_uci_from_move(move), best_piece)
                # print(selected, piece_color_diffs[best_piece_flipped], piece_color_diffs[selected])

                # The other possible piece that needed to be checked was the flipped piece of the best piece,
                # so color difference must be used to compare them
                if best_piece != EMPTY and best_piece_flipped in piece_checks:
                    # print(best_piece, piece_color_diffs[best_piece], piece_color_diffs[best_piece_flipped])
                    if piece_color_diffs[best_piece_flipped] == -1:
                        piece_color_diffs[best_piece_flipped] = self.get_piece_diffs((x, y), best_piece_flipped,
                                                                                     image_grayscale)[1]

                    best_piece = best_piece if piece_color_diffs[best_piece] < \
                                               piece_color_diffs[best_piece_flipped] else best_piece_flipped

                # print(get_uci_from_move(move), best_piece, occupied)

                # occupied is still in the same square, this move has not been made
                if best_piece == occupied:
                    continue

                if best_diff > lowest_diff_move_score:
                    lowest_diff_move_score = best_diff
                    lowest_diff_move = move

        if lowest_diff_move == NO_MOVE:
            # print("LOWEST_DIFF_MOVE IS 0")
            return NO_MOVE

        return lowest_diff_move

    def continuous_detection_handler(self, main_state, engine):

        listener = pynput.mouse.Listener(on_click=on_click)
        listener.start()
        current_mouse_pos = pyautogui.position()

        while main_state.continuous_detection:
            new_mouse_pos = pyautogui.position()

            # print(mouse_down)
            if new_mouse_pos != current_mouse_pos or mouse_down:
                current_mouse_pos = new_mouse_pos
                time.sleep(0.01)
                continue

            current_mouse_pos = new_mouse_pos

            # start = time.time()
            move = self.detect_move_change(main_state)
            # print("CONTINUOUS DETECTION TIME:", time.time() - start)

            if move == NO_MOVE:
                # print("NO MOVE DETECTED")
                continue

            print("MOVE DETECTED: ", get_uci_from_move(move))

            success = main_state.make_move_graphical(move)

            if success and not engine.stopped:
                engine.start_analysis()
