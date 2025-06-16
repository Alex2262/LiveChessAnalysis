import math
import sys
import threading

from objects import *
from game_handler import GameState
from engine_handler import Engine
from detection import Detector
from move import *


def main():
    pygame.init()

    screen_size = DEFAULT_SCREEN_SIZE
    screen = pygame.display.set_mode(screen_size, pygame.RESIZABLE)

    screen.fill(SCREEN_COLOR)

    pygame.display.set_caption("Chess")

    new_mode = 0

    while True:
        if new_mode == -1:
            return
        elif new_mode == 0:
            new_mode = main_menu(screen)


def main_menu(screen):

    engine_directory = os.listdir(FILE_PATH + "engines/")

    engine_names = [x for x in engine_directory]
    engine_files = [FILE_PATH + "engines/" + x for x in engine_directory]

    # Important Variables
    main_state = GameState()
    pv_state = GameState()

    # Detection
    detector = Detector()

    detection_stage = 0  # Nothing has been detected yet.

    screen_size = screen.get_size()

    analysis_button = RectTextButton(LAYER_COLORS[5], (166, DETECT_BACKGROUND_RECT_2[1] + 42, 116, 30), False, 0, 0,
                       "toggle:analysis", "Analysis Off", LAYER_COLORS[6])
    engine_switcher = RectTextButton(LAYER_COLORS[5], (292, DETECT_BACKGROUND_RECT_2[1] + 42, 116, 30), False, 0, 0,
                       "toggle:engines", engine_names[0], LAYER_COLORS[6])

    cont_detect_button = RectTextButton(LAYER_COLORS[5], (216, DETECT_BACKGROUND_RECT_2[1] + 6, 192, 30), False, 0, 0,
                       "toggle:continuous_detection", "Continuous Detection Off", LAYER_COLORS[6], 1.2)

    buttons = [
        RectTextButton(LAYER_COLORS[5], (BOARD_PADDING, DETECT_BACKGROUND_RECT[1] + 6, 120, 30), False, 0, 0,
                       "all:detect_board", "Detect Board", LAYER_COLORS[6]),
        RectTextButton(LAYER_COLORS[5], (BOARD_PADDING, DETECT_BACKGROUND_RECT[1] + 42, 120, 30), False, 0, 0,
                       "all:detect_pieces", "Detect Pieces", LAYER_COLORS[6]),
        RectTextButton(LAYER_COLORS[5], (BOARD_PADDING, DETECT_BACKGROUND_RECT_2[1] + 6, 192, 30), False, 0, 0,
                       "all:detect_new", "Detect New Position", LAYER_COLORS[6]),
        cont_detect_button,
        RectTextButton(WHITE_PERSPECTIVE_COLOR, (124, DETECT_BACKGROUND_RECT_2[1] + 42, 30, 30), True, 0, 0,
                       "toggle:perspective"),
        analysis_button,
        engine_switcher
    ]

    # eval_bar = EvalBar(LAYER2_COLOR, (20, 20, 40, 500), 0, 8)

    basic_objects = [
        # eval_bar,
        RectObject(LAYER_COLORS[3], BOARD_BACKGROUND_RECT, False, 0, 0),
        RectObject(LAYER_COLORS[3], DETECT_BACKGROUND_RECT, False, 0, 0),
        RectObject(LAYER_COLORS[3], DETECT_BACKGROUND_RECT_2, False, 0, 0),
        RectObject(LAYER_COLORS[3], ENGINE_BACKGROUND_RECT_2, False, 0, 0),
        RectTextObject(LAYER_COLORS[4], (BOARD_PADDING, DETECT_BACKGROUND_RECT_2[1] + 42, 100, 30), False, 0, 0,
                       "Perspective: ", LAYER_COLORS[6], 1.3)
    ]

    detect_board_rect = RectTextObject(LAYER_COLORS[4], (132, DETECT_BACKGROUND_RECT[1] + 6, 276, 30), False, 0, 0,
                                       "Board Undetected", LAYER_COLORS[6], 0.7)
    detect_pieces_rect = RectTextObject(LAYER_COLORS[4], (132, DETECT_BACKGROUND_RECT[1] + 42, 276, 30), False, 0, 0,
                                       "Pieces Undetected", LAYER_COLORS[6], 0.7)

    basic_objects.append(detect_board_rect)
    basic_objects.append(detect_pieces_rect)

    # Board Setup
    main_board_gui = Board(CHESS_BOARD_COLOR, (MAIN_BOARD_STARTING_SQUARE[0], MAIN_BOARD_STARTING_SQUARE[1],
                                               MAIN_BOARD_SIZE[0], MAIN_BOARD_SIZE[1]))

    pv_board_gui = Board(CHESS_BOARD_COLOR, (PV_BOARD_STARTING_SQUARE[0], PV_BOARD_STARTING_SQUARE[1],
                                             PV_BOARD_SIZE[0], PV_BOARD_SIZE[1]))

    main_state.board_starting_square = MAIN_BOARD_STARTING_SQUARE
    main_state.default_square_size = MAIN_BOARD_SIZE[0] // 8

    pv_state.board_starting_square = PV_BOARD_STARTING_SQUARE
    pv_state.default_square_size = PV_BOARD_SIZE[0] // 8

    # Engine

    current_engine = 0
    num_engines = len(engine_files)

    starting_engine_file = engine_files[0]

    if PLATFORM == "Windows":
        starting_engine_file = FILE_PATH + "engines/Altair3.0.0_windows_64.exe"
    elif PLATFORM != "Darwin":
        print(PLATFORM + " NOT SUPPORTED")
        return -1

    analysis = False

    name_panel = RectTextObject(LAYER_COLORS[5], (BOARD_PADDING, ENGINE_BACKGROUND_RECT_2[1] + 6, 192, 20),
                                False, 0, 0, "Engine: 0", LAYER_COLORS[6])
    author_panel = RectTextObject(LAYER_COLORS[5], (BOARD_PADDING, ENGINE_BACKGROUND_RECT_2[1] + 32, 192, 20),
                                  False, 0, 0, "Author: 0", LAYER_COLORS[6])
    depth_panel = RectTextObject(LAYER_COLORS[5], (BOARD_PADDING, ENGINE_BACKGROUND_RECT_2[1] + 58, 192, 20),
                                 False, 0, 0, "Depth: 0", LAYER_COLORS[6])
    score_panel = RectTextObject(LAYER_COLORS[5], (BOARD_PADDING, ENGINE_BACKGROUND_RECT_2[1] + 84, 192, 20),
                                 False, 0, 0, "Score: 0", LAYER_COLORS[6])
    nodes_panel = RectTextObject(LAYER_COLORS[5], (BOARD_PADDING, ENGINE_BACKGROUND_RECT_2[1] + 110, 192, 20),
                                 False, 0, 0, "Nodes: 0", LAYER_COLORS[6])
    pv_panel = RectTextObject(LAYER_COLORS[5], (BOARD_PADDING, ENGINE_BACKGROUND_RECT_2[1] + 136, 192, 20),
                              False, 0, 0, "PV: ", LAYER_COLORS[6])

    analysis_moves = []

    basic_objects.append(name_panel)
    basic_objects.append(author_panel)
    basic_objects.append(depth_panel)
    basic_objects.append(score_panel)
    basic_objects.append(nodes_panel)
    basic_objects.append(pv_panel)

    engine = Engine(main_state, starting_engine_file)
    engine_connection_thread = None
    continuous_detection_thread = None

    # Chess
    main_state.initialize_pieces()
    pv_state.initialize_pieces()

    main_state.sprite_group = pygame.sprite.Group()
    main_state.sprite_group.add(main_state.pieces)

    pv_state.sprite_group = pygame.sprite.Group()
    pv_state.sprite_group.add(pv_state.pieces)

    # Pygame Setup

    scale_objects(screen_size, basic_objects, buttons, [main_board_gui, pv_board_gui],
                  main_state.pieces, pv_state.pieces)

    clock = pygame.time.Clock()

    # Main Game Loop

    running = True
    while running:

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False
                engine.stop()

            if event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                screen_size = screen.get_size()
                scale_objects(screen_size, basic_objects, buttons, [main_board_gui, pv_board_gui],
                              main_state.pieces, pv_state.pieces)

            if event.type == pygame.MOUSEBUTTONDOWN:
                pass

            if event.type == pygame.MOUSEBUTTONUP:

                if event.button == 1:  # Left Button
                    mouse_pos = pygame.mouse.get_pos()

                    selected_object = get_selected_object(mouse_pos, buttons)

                    if selected_object is None:
                        continue

                    action = selected_object.get_action().split(":")

                    if action[0] == "toggle":
                        if action[1] == "analysis":
                            analysis = not analysis

                            if analysis:
                                selected_object.text = "Analysis On"
                                engine_connection_thread = threading.Thread(target=engine.connect, args=())
                                engine_connection_thread.start()
                            else:
                                selected_object.text = "Analysis Off"
                                # analysis_moves = []

                                engine.stop()
                                if engine_connection_thread is not None:
                                    engine_connection_thread.join()
                                    engine_connection_thread = None

                            selected_object.update_text()

                        if action[1] == "engines":
                            current_engine = (current_engine + 1) % num_engines
                            selected_object.text = engine_names[current_engine]
                            selected_object.update_text()
                            engine.engine_file = engine_files[current_engine]

                            if analysis:
                                engine.stop()
                                if engine_connection_thread is not None:
                                    engine_connection_thread.join()

                                engine_connection_thread = threading.Thread(target=engine.connect, args=())
                                engine_connection_thread.start()

                        if action[1] == "continuous_detection":
                            main_state.continuous_detection = not main_state.continuous_detection

                            if main_state.continuous_detection:
                                selected_object.text = "Continuous Detection On"

                                continuous_detection_thread = threading.Thread(
                                    target=detector.continuous_detection_handler,
                                    args=(main_state, engine))
                                continuous_detection_thread.start()
                            else:
                                selected_object.text = "Continuous Detection Off"
                                if continuous_detection_thread is not None:
                                    continuous_detection_thread.join()
                                    continuous_detection_thread = None

                            selected_object.update_text()

                        if action[1] == "perspective":
                            main_state.perspective ^= 1
                            main_state.set_perspectives()
                            selected_object.color = WHITE_PERSPECTIVE_COLOR \
                                if main_state.perspective == WHITE_COLOR else BLACK_PERSPECTIVE_COLOR

                    if action[0] == "all":
                        if action[1] == "detect_board" and detection_stage >= 0:
                            print("Detecting Board")

                            screen.fill(SCREEN_COLOR)

                            selected_object = get_selected_object(mouse_pos, buttons)
                            draw_main_objects(screen, screen_size, selected_object, basic_objects, buttons,
                                              [])

                            clock.tick(60)
                            pygame.display.update()

                            retval = detector.get_board_coordinates()

                            detection_stage = 1 if retval else 0

                            detect_board_rect.text = "Success" if retval else "Failed"
                            detect_board_rect.update_text()

                        if action[1] == "detect_pieces" and detection_stage >= 1:
                            print("Detecting Pieces")
                            retval = detector.get_pieces()

                            detection_stage = 2 if retval else 1

                            detect_pieces_rect.text = "Success" if retval else "Failed"
                            detect_pieces_rect.update_text()

                        if action[1] == "detect_new" and detection_stage >= 1:

                            analysis_moves = []

                            if analysis:
                                engine.stop()

                            print("Detecting New")
                            new_fen = detector.get_new_basic_fen(main_state) + " "
                            new_fen += "w" if main_state.perspective == WHITE_COLOR else "b"
                            test_fen = new_fen + " - - 0 1"

                            main_state.position.parse_fen(test_fen)

                            castling_flag = " "

                            if main_state.position.board[E1] == WHITE_KING:
                                if main_state.position.board[H1] == WHITE_ROOK:
                                    castling_flag += "K"
                                if main_state.position.board[A1] == WHITE_ROOK:
                                    castling_flag += "Q"

                            if main_state.position.board[E8] == BLACK_KING:
                                if main_state.position.board[H8] == BLACK_ROOK:
                                    castling_flag += "k"
                                if main_state.position.board[A8] == BLACK_ROOK:
                                    castling_flag += "q"

                            if castling_flag == " ":
                                castling_flag = " -"

                            new_fen += castling_flag + " - 0 1"

                            main_state.fen = new_fen
                            main_state.initialize_pieces()

                            if analysis:
                                engine.start_analysis()

                            main_state.sprite_group = pygame.sprite.Group()
                            main_state.sprite_group.add(main_state.pieces)

                            scale_objects(screen_size, basic_objects, buttons, [main_board_gui, pv_board_gui],
                                          main_state.pieces, pv_state.pieces)
                            # pv_state.sprite_group.add(pv_state.pieces)

        mouse_pos = pygame.mouse.get_pos()

        if analysis:
            if engine_connection_thread is not None and engine.connection_successful:
                engine_connection_thread.join()
                engine_connection_thread = None

                engine.start_analysis()

            if engine.info["depth"] >= 1:
                name_panel.text = "Engine: " + str(engine.info["name"])
                author_panel.text = "Author: " + str(engine.info["author"])
                depth_panel.text = "Depth: " + str(engine.info["depth"])

                score_info = engine.info["evaluation"] * (-1 if main_state.position.side == 1 else 1)
                if engine.info["evaluation_type"] == "cp":
                    score_info = str(score_info / 100.0)
                else:
                    score_info = ("M" if score_info >= 0 else "-M") + str(abs(score_info))

                score_panel.text = "Score: " + score_info
                nodes_panel.text = "Nodes: " + str(engine.info["nodes"])
                pv_panel.text = "PV: " + " ".join(engine.info["pv"].split()[:5])

                name_panel.update_text()
                author_panel.update_text()
                depth_panel.update_text()
                score_panel.update_text()
                nodes_panel.update_text()
                pv_panel.update_text()

                engine_move_coordinates = []
                uci_move = get_move_from_uci(main_state.position, engine.info["pv"].split()[0])
                analysis_moves = [(uci_move, engine.info["evaluation"])]

                origin_square = MAILBOX_TO_STANDARD[get_from_square(uci_move)]
                target_square = MAILBOX_TO_STANDARD[get_to_square(uci_move)]

                # eval_bar.update_evaluation(engine.info["evaluation"] * (-1 if main_state.position.side == 1 else 1),
                #                            engine.info["evaluation_type"])

        if continuous_detection_thread is not None and detector.interrupt_cont_detect:
            detector.interrupt_cont_detect = False
            cont_detect_button.text = "Continuous Detection Off"
            continuous_detection_thread.join()
            continuous_detection_thread = None

        if len(main_state.current_moves) == 0:
            print("PLAYER LOST")

            # main_state.in_play = False
            analysis = False
            analysis_button.text = "Analysis Off"
            engine.stop()

        screen.fill(SCREEN_COLOR)

        selected_object = get_selected_object(mouse_pos, buttons)
        draw_main_objects(screen, screen_size, selected_object, basic_objects, buttons,
                          [main_board_gui, pv_board_gui])
        main_state.sprite_group.draw(screen)
        draw_analysis_moves(screen, main_state, main_board_gui, analysis_moves)

        clock.tick(60)
        pygame.display.update()

    engine.stop()
    if continuous_detection_thread is not None:
        continuous_detection_thread.join()

    pygame.display.quit()
    pygame.quit()
    sys.exit()


def get_selected_object(mouse_pos, buttons):
    for button in buttons:
        if button.is_selecting(mouse_pos):
            return button
    return None


def draw_basic_objects(surface, objects):
    for basic_object in objects:
        basic_object.draw(surface, False)


def draw_buttons(surface, selected_object, buttons):
    for button in buttons:
        if button == selected_object:
            button.draw(surface, True)
        else:
            button.draw(surface, False)


def draw_main_objects(screen, screen_size, selected_object, basic_objects, buttons, boards):

    surface1 = pygame.Surface(screen_size, pygame.SRCALPHA)
    draw_basic_objects(surface1, basic_objects)

    for board in boards:
        board.draw(surface1, False)

    surface2 = pygame.Surface(screen_size, pygame.SRCALPHA)
    draw_buttons(surface2, selected_object, buttons)

    surface1.blit(surface2, (0, 0))
    screen.blit(surface1, (0, 0))


def draw_arrow(screen, start, end, color, thickness, sq_size):

    y_length = end[1] - start[1]
    x_length = end[0] - start[0]

    length_ratio = math.sqrt(x_length ** 2 + y_length ** 2) / sq_size

    error = 1000000
    line_slope = error if x_length == 0 else y_length / x_length
    perpendicular = error if line_slope == 0 else -(1 / line_slope)

    triangle_ratio = 0.3

    # Ensure nice looking triangle proportional
    side_ratio = triangle_ratio * (2 ** 0.5) / length_ratio

    optimal_size = sq_size * triangle_ratio

    # Calculate the intersection of a line (y = mx) and circle (x^2 + y^2 = optimal_size)
    # where center is assumed at (0, 0) for simplicity of calculations
    triangle_side_x = optimal_size if line_slope == error else math.sqrt(
        abs((optimal_size ** 2) / (perpendicular ** 2 + 1))
    )

    # Get the Y length based on X length
    triangle_side_y = optimal_size if perpendicular == error else triangle_side_x * perpendicular

    # Calculate the point which will be the midpoint of the base of the triangle
    point = (end[0] - int(end[0] - start[0]) * side_ratio,
             end[1] - int(end[1] - start[1]) * side_ratio)

    triangle = (end,

                # Vertices from midpoint of the base of the triangle
                (point[0] + triangle_side_x, point[1] + triangle_side_y),
                (point[0] - triangle_side_x, point[1] - triangle_side_y))

    pygame.draw.line(screen, color, start, point, thickness)  # Draw line
    pygame.draw.polygon(screen, color, triangle, 0)  # Draw triangle


def draw_analysis_moves(screen, main_state, board, analysis_moves):
    if len(analysis_moves) == 0:
        return

    new_surface = pygame.Surface((screen.get_width(), screen.get_height()), pygame.SRCALPHA)

    best_evaluation = analysis_moves[0][1]

    for move, evaluation in analysis_moves:
        origin_square = MAILBOX_TO_STANDARD[get_from_square(move)] ^ (63 * main_state.perspective)
        target_square = MAILBOX_TO_STANDARD[get_to_square(move)] ^ (63 * main_state.perspective)

        origin = (board.x + (origin_square % 8) * board.sq_size + board.sq_size // 2,
                  board.y + (origin_square // 8) * board.sq_size + board.sq_size // 2)
        target = (board.x + (target_square % 8) * board.sq_size + board.sq_size // 2,
                  board.y + (target_square // 8) * board.sq_size + board.sq_size // 2)

        eval_diff = abs(best_evaluation - evaluation)

        if eval_diff <= 4 or evaluation >= 800:
            rank = 0
        elif eval_diff <= 15 or evaluation >= 500:
            rank = 1
        elif eval_diff <= 45 or evaluation >= 300:
            rank = 2
        elif eval_diff <= 95 or evaluation >= 175:
            rank = 3
        elif eval_diff <= 180 or evaluation >= 50:
            rank = 4
        else:
            rank = 5

        draw_arrow(new_surface, origin, target, ARROW_COLORS[rank], board.sq_size // 5, board.sq_size)

    screen.blit(new_surface, (0, 0))


def scale_objects(screen_size, *objects):
    for object_list in objects:
        for basic_object in object_list:
            basic_object.scale(screen_size)


if __name__ == "__main__":
    main()
