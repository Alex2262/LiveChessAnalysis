
import numpy as np
import os
import platform

PLATFORM = platform.system()
FILE_PATH = os.path.dirname(__file__) + "/../"

'''


GUI


'''
DEFAULT_SCREEN_SIZE = (420, 700)

BOARD_PADDING = 12
MAIN_BOARD_STARTING_SQUARE = (BOARD_PADDING, 2 * BOARD_PADDING)

MAIN_BOARD_SIZE = (DEFAULT_SCREEN_SIZE[0] // 2 - int(1.5 * BOARD_PADDING),
                   DEFAULT_SCREEN_SIZE[0] // 2 - int(1.5 * BOARD_PADDING))

PV_BOARD_STARTING_SQUARE = (BOARD_PADDING // 2 + DEFAULT_SCREEN_SIZE[0] // 2, 2 * BOARD_PADDING)
PV_BOARD_SIZE = (DEFAULT_SCREEN_SIZE[0] // 2 - int(1.5 * BOARD_PADDING),
                 DEFAULT_SCREEN_SIZE[0] // 2 - int(1.5 * BOARD_PADDING))

BOARD_BACKGROUND_RECT = (0, MAIN_BOARD_STARTING_SQUARE[1] - BOARD_PADDING,
                         DEFAULT_SCREEN_SIZE[0], MAIN_BOARD_SIZE[1] + 2 * BOARD_PADDING)

DETECT_BACKGROUND_RECT = (0, BOARD_BACKGROUND_RECT[1] + BOARD_BACKGROUND_RECT[3] + BOARD_PADDING,
                          DEFAULT_SCREEN_SIZE[0], 78)

DETECT_BACKGROUND_RECT_2 = (0, DETECT_BACKGROUND_RECT[1] + DETECT_BACKGROUND_RECT[3] + BOARD_PADDING,
                            DEFAULT_SCREEN_SIZE[0], 78)

ENGINE_BACKGROUND_RECT_2 = (0, DETECT_BACKGROUND_RECT_2[1] + DETECT_BACKGROUND_RECT_2[3] + BOARD_PADDING,
                            DEFAULT_SCREEN_SIZE[0], 162)

LAYER_COLORS = [
    (36, 42, 54),   # NORD DARK 1
    (46, 52, 64),   # NORD DARK 2
    (59, 66, 82),   # NORD DARK 3
    (62, 71, 89),   # NORD DARK 4
    (67, 76, 94),   # NORD DARK 5
    (76, 86, 106),  # NORD DARK 6
    (216, 222, 233),  # NORD LIGHT 1
    (229, 233, 240),  # NORD LIGHT 2
    (236, 239, 244),  # NORD LIGHT 3
    (94, 129, 172),      # NORD MID BLUE 1
    (129, 161, 193),     # NORD MID BLUE 2
]

SCREEN_COLOR = LAYER_COLORS[5]

LAYER1_COLOR = (122, 114, 101)
LAYER2_COLOR = (67, 62, 63)  # this can be added
LAYER3_COLOR = (94, 89, 81)
LAYER4_COLOR = (192, 183, 177)

WHITE_PERSPECTIVE_COLOR = (236, 239, 244)
BLACK_PERSPECTIVE_COLOR = (19, 16, 11)

BUTTON1_COLOR = (192, 183, 177)
BUTTON2_COLOR = ()  # this can be added

TEXT_COLOR = (20, 20, 20)
BUTTON_TEXT_COLOR = (20, 20, 20)

#                      LIGHT COLOR      DARK COLOR
CHESS_BOARD_COLOR = ((198, 156, 114), (142, 110, 83))

ARROW_COLORS = ((3, 44, 252, 100), (157, 252, 3, 40), (252, 219, 3, 40),
                (252, 144, 3, 40), (252, 40, 3, 40))

'''


DETECTION


'''

PIECE_NAMES_INDEX = {
    '0': ('white_pawn', (6, 0)),
    '1': ('white_knight', (7, 1)),
    '2': ('white_bishop', (7, 2)),
    '3': ('white_rook', (7, 0)),
    '4': ('white_queen', (7, 3)),
    '5': ('white_king', (7, 4)),
    '6': ('black_pawn', (1, 0)),
    '7': ('black_knight', (0, 1)),
    '8': ('black_bishop', (0, 2)),
    '9': ('black_rook', (0, 0)),
    '10': ('black_queen', (0, 3)),
    '11': ('black_king', (0, 4)),
}

PIECE_NAMES_FEN = {
    'white_pawn': 'P',
    'white_knight': 'N',
    'white_bishop': 'B',
    'white_rook': 'R',
    'white_queen': 'Q',
    'white_king': 'K',
    'black_pawn': 'p',
    'black_knight': 'n',
    'black_bishop': 'b',
    'black_rook': 'r',
    'black_queen': 'q',
    'black_king': 'k',
}

'''



POSITION | CHESS BACKEND



'''

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Piece/Move/Position Constants
WHITE_COLOR = 0
BLACK_COLOR = 1

MOVE_TYPE_NORMAL    = 0
MOVE_TYPE_EP        = 1
MOVE_TYPE_CASTLE    = 2
MOVE_TYPE_PROMOTION = 3

WHITE_PAWN        = 0
WHITE_KNIGHT      = 1
WHITE_BISHOP      = 2
WHITE_ROOK        = 3
WHITE_QUEEN       = 4
WHITE_KING        = 5

BLACK_PAWN        = 6
BLACK_KNIGHT      = 7
BLACK_BISHOP      = 8
BLACK_ROOK        = 9
BLACK_QUEEN       = 10
BLACK_KING        = 11

EMPTY           = 12
PADDING         = 13

NO_MOVE = 0

A1 = 91
A8 = 21
H1 = 98
H8 = 28

E1 = 95
E8 = 25
C1 = 93
C8 = 23
G1 = 97
G8 = 27


# Translations from 12x10 array to 8x8 and back
STANDARD_TO_MAILBOX = np.array((
    21, 22, 23, 24, 25, 26, 27, 28,
    31, 32, 33, 34, 35, 36, 37, 38,
    41, 42, 43, 44, 45, 46, 47, 48,
    51, 52, 53, 54, 55, 56, 57, 58,
    61, 62, 63, 64, 65, 66, 67, 68,
    71, 72, 73, 74, 75, 76, 77, 78,
    81, 82, 83, 84, 85, 86, 87, 88,
    91, 92, 93, 94, 95, 96, 97, 98
))

MAILBOX_TO_STANDARD = np.array((
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99,  0,  1,  2,  3,  4,  5,  6,  7, 99,
    99,  8,  9, 10, 11, 12, 13, 14, 15, 99,
    99, 16, 17, 18, 19, 20, 21, 22, 23, 99,
    99, 24, 25, 26, 27, 28, 29, 30, 31, 99,
    99, 32, 33, 34, 35, 36, 37, 38, 39, 99,
    99, 40, 41, 42, 43, 44, 45, 46, 47, 99,
    99, 48, 49, 50, 51, 52, 53, 54, 55, 99,
    99, 56, 57, 58, 59, 60, 61, 62, 63, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99
))

# Matching the piece index to a respective char
PIECE_MATCHER = [
    'P',
    'N',
    'B',
    'R',
    'Q',
    'K',
]

# Increments for traversing a board
WHITE_INCREMENTS = np.array((
    (-11,  -9, -10, -20,   0,   0,   0,   0),  # Pawn
    (-21, -19,  -8,  12,  21,  19,   8, -12),  # Knight
    (-11,  11,   9,  -9,   0,   0,   0,   0),
    (-10,   1,  10,  -1,   0,   0,   0,   0),
    (-11,  11,   9,  -9, -10,   1,  10,  -1),
    (-11, -10,  -9,   1,  11,  10,   9,  -1)
))

BLACK_INCREMENTS = np.array((
    ( 11,   9,  10,  20,   0,   0,   0,   0),
    (-21, -19,  -8,  12,  21,  19,   8, -12),
    (-11,  11,   9,  -9,   0,   0,   0,   0),
    (-10,   1,  10,  -1,   0,   0,   0,   0),
    (-11,  11,   9,  -9, -10,   1,  10,  -1),
    (-11, -10,  -9,   1,  11,  10,   9,  -1)
))

ATK_INCREMENTS = np.array((
    (-11,  -9,   0,   0,   0,   0,   0,   0),
    (-21, -19,  -8,  12,  21,  19,   8, -12),
    (-11,  11,   9,  -9,   0,   0,   0,   0),
    (-10,   1,  10,  -1,   0,   0,   0,   0),
    (-11,  11,   9,  -9, -10,   1,  10,  -1),
    (-11, -10,  -9,   1,  11,  10,   9,  -1)
))
