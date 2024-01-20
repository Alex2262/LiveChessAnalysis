
import pygame

from position import Position
from objects import Piece
from move import *


class GameState:
    def __init__(self):

        self.board_starting_square = ()
        self.default_square_size = 0

        self.position = Position()
        self.pieces = []
        self.piece_images = []
        self.fen = START_FEN
        self.perspective = WHITE_COLOR

        self.sprite_group = None

        self.in_play = True

        self.old_ep_square = 0
        self.old_castle_ability_bits = 0

        self.continuous_detection = False

        self.move_archive = []

        self.current_moves = self.position.get_legal_moves()

        self.initialize_piece_images()

    def initialize_piece_images(self):
        color_names = ["w", "b"]
        for color in range(2):
            new_arr = []
            for piece in range(6):
                new_arr.append(pygame.image.load(
                    FILE_PATH + "images/{}.png".format(color_names[color] + PIECE_MATCHER[piece])).convert_alpha())
            self.piece_images.append(new_arr)

    def initialize_pieces(self):
        self.pieces = []
        self.move_archive = []

        self.position.parse_fen(self.fen)

        for i in range(64):

            pos = STANDARD_TO_MAILBOX[i]
            row = i // 8
            col = i % 8
            piece = self.position.board[pos]
            if piece < EMPTY:
                self.pieces.append(Piece((self.board_starting_square[0] + self.default_square_size * col,
                                                self.board_starting_square[1] + self.default_square_size * row,
                                                self.default_square_size,
                                                self.default_square_size),
                                        self.board_starting_square,
                                        row, col, piece >= 6,
                                        PIECE_MATCHER[self.position.board[pos] % 6].lower(),
                                        self.piece_images))

                self.pieces[-1].perspective = self.perspective

    def set_perspectives(self):
        for piece in self.pieces:
            piece.set_perspective(self.perspective)

    def get_piece(self, col, row):
        for piece in self.pieces:
            if piece.col == col and piece.row == row:
                return piece

        return None

    def make_move(self, move):
        self.old_ep_square = self.position.ep_square
        self.old_castle_ability_bits = self.position.castle_ability_bits

        self.position.make_move(move)
        self.position.side ^= 1

        self.move_archive.append(move)
        self.current_moves = self.position.get_legal_moves()

    def make_move_graphical(self, move):

        origin_square = get_from_square(move)
        origin_square_standard = MAILBOX_TO_STANDARD[origin_square]

        target_square = get_to_square(move)
        target_square_standard = MAILBOX_TO_STANDARD[target_square]

        origin_col = origin_square_standard % 8
        origin_row = origin_square_standard // 8

        target_col = target_square_standard % 8
        target_row = target_square_standard // 8

        selected_piece = self.get_piece(origin_col, origin_row)
        removed_piece  = self.get_piece(target_col, target_row)

        # print(origin_col, origin_row, selected_piece)
        if selected_piece is None:
            return False

        promotion_piece = get_promotion_piece(move)

        if get_move_type(move) == MOVE_TYPE_EP:
            removed_piece = self.get_piece(target_col, target_row + (1 if self.position.side == 0 else -1))

        if removed_piece is not None:
            self.pieces.remove(removed_piece)
            self.sprite_group.remove(removed_piece)

        selected_piece.move(target_col, target_row)

        if get_move_type(move) == MOVE_TYPE_PROMOTION:
            selected_piece.piece = PIECE_MATCHER[promotion_piece]
            selected_piece.image = pygame.Surface((selected_piece.width, selected_piece.height),
                                                  pygame.SRCALPHA).convert_alpha()
            selected_piece.unedited_sprite = pygame.image.load(
                FILE_PATH + "images/{}.png".format(selected_piece.color + selected_piece.piece)).convert_alpha()

            selected_piece.sprite = pygame.transform.smoothscale(
                selected_piece.unedited_sprite,
                (selected_piece.width, selected_piece.height))

            selected_piece.image.blit(selected_piece.sprite, (0, 0))

        if get_move_type(move) == MOVE_TYPE_CASTLE:
            rook_col = 0 if selected_piece.col == 2 else 7
            rook_row = target_row

            new_rook_col = 3 if rook_col == 0 else 5

            rook = self.get_piece(rook_col, rook_row)
            rook.move(new_rook_col, rook_row)

        self.make_move(move)

        return True

    def undo_move(self):
        if len(self.move_archive) == 0 or self.move_archive[-1] == 0:
            print("INVALID LAST MOVE")
            return

        self.position.undo_move(self.move_archive[-1], self.old_ep_square, self.old_castle_ability_bits)
        self.position.side ^= 1

        self.old_ep_square = 0
        self.old_castle_ability_bits = 0
        self.move_archive.pop(-1)
