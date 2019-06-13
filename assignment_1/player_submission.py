#!/usr/bin/env python
from isolation import Board, game_as_text
from random import randint


# This file is your main submission that will be graded against. Do not
# add any classes or functions to this file that are not part of the classes
# that we want.

# Submission Class 1

class OpenMoveEvalFn:

    def score(self, game, maximizing_player_turn=True):
        """Score the current game state
        
        Evaluation function that outputs a score equal to how many 
        moves are open for AI player on the board.
            
        Args
            param1 (Board): The board and game state.
            param2 (bool): True if maximizing player is active.

        Returns:
            float: The current state's score. Number of your agent's moves.
            
        """
	
        # TODO: finish this function!
        #raise NotImplementedError

        numMove = len(game.get_legal_moves())
        if maximizing_player_turn and numMove == 0:
            # max player can't move, max lose
            eval_func = float("-inf")
        # elif maximizing_player_turn and numMove == 1:
        #     # min player can't move, max win
        #     eval_func = float("inf")
        elif not maximizing_player_turn and numMove == 0:
            # min player can't move, max win
            eval_func = float("inf")
        # elif not maximizing_player_turn and numMove == 1:
        #     # max player can't move, min win
        #     eval_func = float("-inf")
        else:
            eval_func = numMove
        return eval_func

# Submission Class 2
class CustomEvalFn:

    def __init__(self):
        pass

    def move_diagonal_occupied(self,game,move):
        row, col = move

        if row > 0 and col > 0 and game.__board_state__[row - 1][col - 1] == '0':
            return False
        if row > 0 and col < 6 and game.__board_state__[row - 1][col + 1] == '0':
            return False
        if row < 6 and col > 0 and game.__board_state__[row + 1][col - 1] == '0':
            return False
        if row < 6 and col < 6 and game.__board_state__[row + 1][col - 1]== '0':
            return False

        return True

    def score(self, game, maximizing_player_turn=True):
        """Score the current game state
        
        Custom evaluation function that acts however you think it should. This 
        is not required but highly encouraged if you want to build the best 
        AI possible.
        
        Args
            game (Board): The board and game state.
            maximizing_player_turn (bool): True if maximizing player is active.

        Returns:
            float: The current state's score, based on your own heuristic.
            
        """
        # TODO: finish this function!
        legalMoves=game.get_legal_moves()
        numMove=len(legalMoves)
        eval_func = numMove
        if maximizing_player_turn and numMove==0:
            # max player can't move, max lose
            eval_func=float("-inf")
        elif  maximizing_player_turn and numMove==1:
            if self.move_diagonal_occupied(game, legalMoves[0]):
            # min player can't move, max win
                eval_func = float("inf")
        elif not maximizing_player_turn and numMove == 0:
                # max player can't move, max lose
                eval_func = float("inf")
        elif not maximizing_player_turn and numMove==1:
            # min player can't move, max win
            if  self.move_diagonal_occupied(game, legalMoves[0]):
                eval_func = float("-inf")
        #raise NotImplementedError
        return eval_func



class CustomPlayer:
    # TODO: finish this class!
    """Player that chooses a move using 
    your evaluation function and 
    a minimax algorithm 
    with alpha-beta pruning.
    You must finish and test this player
    to make sure it properly uses minimax
    and alpha-beta to return a good move."""

    def __init__(self, search_depth=49, eval_fn=CustomEvalFn()):
        """Initializes your player.
        
        if you find yourself with a superior eval function, update the default 
        value of `eval_fn` to `CustomEvalFn()`
        
        Args:
            search_depth (int): The depth to which your agent will search
            eval_fn (function): Utility function used by your agent
        """
        self.eval_fn = eval_fn
        self.search_depth = search_depth

    def move(self, game, legal_moves, time_left):
        """Called to determine one move by your agent

        Args:
            game (Board): The board and game state.
            legal_moves (dict): Dictionary of legal moves and their outcomes
            time_left (function): Used to determine time left before timeout

        Returns:
            (tuple): best_move
        """

        if len(legal_moves)==0:
            return None
        if len(legal_moves)==1:
            return legal_moves[0]
        if game.move_count == 0:
            return (game.width / 2, game.height / 2)
        if game.move_count == 1 and (game.width / 2, game.height / 2) in legal_moves:
            return (game.width / 2, game.height / 2)
        for move in legal_moves:
            if len(game.forecast_move(move).get_legal_moves())==0:
                return move

        best_move, utility = self.iter_deep(game, time_left, depth=self.search_depth)
        # change minimax to alphabeta,iter_deep after completing alphabeta part of assignment
        #print best_move,utility
        return best_move

    def utility(self, game,maximizing_player):
        """Can be updated if desired"""
        #print maximizing_player
        return self.eval_fn.score(game,maximizing_player)

    def rand_move(self,game):
        legal_moves=game.get_legal_moves()
        num_moves=len( legal_moves)
        return legal_moves[randint(0,num_moves-1)]

    def minimax(self, game, time_left, depth=3, maximizing_player=True):
        """Implementation of the minimax algorithm
                    print time_left

        Args:
            game (Board): A board and game state.
            time_left (function): Used to determine time left before timeout
            depth: Used to track how deep you are in the search tree
            maximizing_player (bool): True if maximizing player is active.

        Returns:
            (tuple, int): best_move, best_val
        """
        # TODO: finish this function!
        eval_score=self.utility(game,maximizing_player)
        legalMoves = game.get_legal_moves()
        if game.move_count == 0 and maximizing_player:
            return (game.width / 2, game.height / 2), game.width / 2 * 8
        if game.move_count == 1 and maximizing_player and (game.width / 2, game.height / 2) in legalMoves:
            return (game.width / 2, game.height / 2), self.utility(game.forecast_move((game.width/2, game.height/2)), not maximizing_player)
        if depth == 0 or time_left()<100 or len(legalMoves)==0 or (legalMoves is None):
            return None, eval_score

        best_move = self.rand_move(game)
        # fore_score = self.utility(game.forecast_move(best_move), not maximizing_player)

        if maximizing_player:
            #print "max"
            best_val = float("-inf")
            for move in legalMoves:
                _, nextScore=self.minimax(game.forecast_move(move),time_left,depth-1,False)
                if nextScore>=best_val:
                    best_move,best_val=move,nextScore
            return best_move, best_val
        else:
            #print "min"
            best_val = float("inf")
            for move in legalMoves:
                _,nextScore=self.minimax(game.forecast_move(move),time_left,depth-1,True)
                if nextScore<=best_val:
                    best_move, best_val = move, nextScore
        #raise NotImplementedError
            return best_move, best_val

    def alphabeta(self, game, time_left, depth=3, alpha=float("-inf"), beta=float("inf"),
                  maximizing_player=True):
        """Implementation of the alphabeta algorithm
        
        Args:
            game (Board): A board and game state.
            time_left (function): Used to determine time left before timeout
            depth: Used to track how deep you are in the search tree
            alpha (float): Alpha value for pruning
            beta (float): Beta value for pruning
            maximizing_player (bool): True if maximizing player is active.

        Returns:
            (tuple, int): best_move, best_val
        """
        # TODO: finish this function!

        eval_score = self.utility(game, maximizing_player)
        legalMoves = game.get_legal_moves()

        threshold = 62
        #print time_left()

        if game.move_count==0 and maximizing_player:
            return (game.width/2, game.height/2),game.width/2*8
        if game.move_count==1 and maximizing_player and (game.width/2, game.height/2) in legalMoves:
            return (game.width/2, game.height/2),self.utility(game.forecast_move((game.width/2, game.height/2)), not maximizing_player)
        if depth == 0 or time_left() < threshold or len(legalMoves) == 0 or legalMoves is None:
            return None, eval_score
        best_move = self.rand_move(game)
        #fore_score = self.utility(game.forecast_move(best_move), not maximizing_player)

        if maximizing_player:
            # print "max"
            val = alpha
            for move in legalMoves:
                _, nextScore = self.alphabeta(game.forecast_move(move), time_left, depth - 1, val,beta, False)
                if nextScore > val:
                    best_move, val = move, nextScore
                if val>beta:
                    return best_move,beta
                #alpha = max(alpha, val)
            return best_move, val
        else:
            val = beta
            for move in legalMoves:
                _, nextScore = self.alphabeta(game.forecast_move(move), time_left, depth - 1, alpha, val, True)
                if nextScore < val:
                    best_move, val = move, nextScore
                if val < alpha:
                    return best_move,alpha
                #beta = min(beta, val)
            return best_move, val

    def iter_deep (self, game, time_left, depth=3, alpha=float("-inf"), beta=float("inf"),
                  maximizing_player=True):

        legalMoves = game.get_legal_moves()
        if game.move_count == 0 and maximizing_player:
            return (game.width / 2, game.height / 2), game.width / 2 * 8
        if game.move_count == 1 and maximizing_player and (game.width / 2, game.height / 2) in legalMoves:
            return (game.width / 2, game.height / 2), self.utility(game.forecast_move((game.width/2, game.height/2)), not maximizing_player)
        if game.move_count>1 and game.move_count<=5 and game.move_count%2==1 and  maximizing_player:
            best_move = self.rand_move(game)
            fore_score = self.utility(game.forecast_move(best_move), not maximizing_player)
            return best_move,fore_score

        best_move = self.rand_move(game)
        fore_score = self.utility(game.forecast_move(best_move), not maximizing_player)

        threshold = 62
        for i in range(6,depth):
            #print "time_left",time_left()
            #print "threshold", threshold
            #print i
            if time_left() <=threshold or fore_score==float("inf"):
                return best_move,fore_score
            move,score=self.alphabeta(game,time_left,i,alpha,beta,maximizing_player)
            if not score==float("-inf"):
                best_move=move
                fore_score=score
                #print best_move,fore_score
        return best_move,fore_score

class CustomPlayer_mini:
        # TODO: finish this class!
        """Player that chooses a move using
        your evaluation function and
        a minimax algorithm
        with alpha-beta pruning.
        You must finish and test this player
        to make sure it properly uses minimax
        and alpha-beta to return a good move."""

        def __init__(self, search_depth=3, eval_fn=OpenMoveEvalFn()):
            """Initializes your player.

            if you find yourself with a superior eval function, update the default
            value of `eval_fn` to `CustomEvalFn()`

            Args:
                search_depth (int): The depth to which your agent will search
                eval_fn (function): Utility function used by your agent
            """
            self.eval_fn = eval_fn
            self.search_depth = search_depth

        def move(self, game, legal_moves, time_left):
            """Called to determine one move by your agent

            Args:
                game (Board): The board and game state.
                legal_moves (dict): Dictionary of legal moves and their outcomes
                time_left (function): Used to determine time left before timeout

            Returns:
                (tuple): best_move
            """

            if len(legal_moves) == 0:
                return None
            elif len(legal_moves) == 1:
                return legal_moves[0]
            else:
                best_move, utility = self.minimax(game, time_left, depth=self.search_depth)
                # change minimax to alphabeta,iter_deep after completing alphabeta part of assignment
                # print best_move,utility
                return best_move

        def utility(self, game, maximizing_player):
            """Can be updated if desired"""
            return self.eval_fn.score(game, maximizing_player)

        def rand_move(self, game):
            legal_moves = game.get_legal_moves()
            num_moves = len(legal_moves)
            return legal_moves[randint(0, num_moves - 1)]

        def minimax(self, game, time_left, depth=3, maximizing_player=True):
            """Implementation of the minimax algorithm
                        print time_left

            Args:
                game (Board): A board and game state.
                time_left (function): Used to determine time left before timeout
                depth: Used to track how deep you are in the search tree
                maximizing_player (bool): True if maximizing player is active.

            Returns:
                (tuple, int): best_move, best_val
            """
            # TODO: finish this function!
            eval_score = self.utility(game, maximizing_player)
            legalMoves = game.get_legal_moves()
            if game.move_count == 0 and maximizing_player:
                return (game.width / 2, game.height / 2), game.width / 2 * 8
            if game.move_count == 1 and maximizing_player and (game.width / 2, game.height / 2) in legalMoves:
                return (game.width / 2, game.height / 2), self.utility(
                    game.forecast_move((game.width / 2, game.height / 2)), not maximizing_player)
            if depth == 0 or time_left() < 1 or len(legalMoves) == 0 or (legalMoves is None):
                return None, eval_score

            best_move = self.rand_move(game)
            # fore_score = self.utility(game.forecast_move(best_move), not maximizing_player)

            if maximizing_player:
                # print "max"
                best_val = float("-inf")
                for move in legalMoves:
                    _, nextScore = self.minimax(game.forecast_move(move), time_left, depth - 1, False)
                    if nextScore >= best_val:
                        best_move, best_val = move, nextScore
                return best_move, best_val
            else:
                # print "min"
                best_val = float("inf")
                for move in legalMoves:
                    _, nextScore = self.minimax(game.forecast_move(move), time_left, depth - 1, True)
                    if nextScore <= best_val:
                        best_move, best_val = move, nextScore
                        # raise NotImplementedError
                return best_move, best_val