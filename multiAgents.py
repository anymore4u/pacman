def scoreEvaluationFunction(currentGameState):
    return currentGameState.getScore()

from util import manhattanDistance
from game import Directions
import random, util
from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining its alternatives via a state evaluation function.
    """
    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.
        """
        # Este método provavelmente já existe e não precisa ser alterado.
        legalMoves = gameState.getLegalActions()
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        # Substitua a função de avaliação existente por esta
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        minFoodDistance = min(foodDistances) if foodDistances else 1

        ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        minGhostDistance = min(ghostDistances) if ghostDistances else 1

        if minGhostDistance < 2 and min(newScaredTimes) == 0:
            ghostPenalty = 1000
        else:
            ghostPenalty = 0

        score = successorGameState.getScore()
        score += 1.0 / minFoodDistance
        score -= ghostPenalty

        return score

    def scoreEvaluationFunction(currentGameState: GameState):
        """
        This default evaluation function just returns the score of the state.
        The score is the same one displayed in the Pacman GUI.

        This evaluation function is meant for use with adversarial search agents
        (not reflex agents).
        """
        return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0  # Pacman is always agent index 0
        if evalFn == 'betterEvaluationFunction':
            self.evaluationFunction = betterEvaluationFunction
        else:
            self.evaluationFunction = scoreEvaluationFunction
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        def minimax(agent, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            if agent == 0:
                return max(minimax(1, depth, gameState.generateSuccessor(agent, action)) for action in gameState.getLegalActions(agent))
            else:
                next_agent = agent + 1 if agent + 1 < gameState.getNumAgents() else 0
                new_depth = depth + 1 if next_agent == 0 else depth
                return min(minimax(next_agent, new_depth, gameState.generateSuccessor(agent, action)) for action in gameState.getLegalActions(agent))

        legalMoves = gameState.getLegalActions(0)
        scores = [minimax(1, 0, gameState.generateSuccessor(0, action)) for action in legalMoves]
        bestMove = max(scores)
        bestIndices = [index for index, value in enumerate(scores) if value == bestMove]
        chosenIndex = random.choice(bestIndices)
        return legalMoves[chosenIndex]

class AlphaBetaAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        def alphaBeta(agent, depth, gameState, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            if agent == 0:  # Pac-Man's turn (Maximizer)
                value = float('-inf')
                for action in gameState.getLegalActions(agent):
                    value = max(value, alphaBeta(1, depth, gameState.generateSuccessor(agent, action), alpha, beta))
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
                return value
            else:  # Ghosts' turn (Minimizer)
                value = float('inf')
                next_agent = agent + 1 if agent + 1 < gameState.getNumAgents() else 0
                new_depth = depth + 1 if next_agent == 0 else depth
                for action in gameState.getLegalActions(agent):
                    value = min(value, alphaBeta(next_agent, new_depth, gameState.generateSuccessor(agent, action), alpha, beta))
                    if value < alpha:
                        return value
                    beta = min(beta, value)
                return value

        # Start from Pac-Man (agent 0) and depth 0
        bestAction = None
        value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        for action in gameState.getLegalActions(0):
            newValue = alphaBeta(1, 0, gameState.generateSuccessor(0, action), alpha, beta)
            if newValue > value:
                value = newValue
                bestAction = action
            alpha = max(alpha, value)
        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        def expectimax(agent, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            if agent == 0:  # Pac-Man's turn (Maximizer)
                return max(expectimax(1, depth, gameState.generateSuccessor(agent, action)) for action in gameState.getLegalActions(agent))
            else:  # Ghosts' turn (Expectation)
                next_agent = agent + 1 if agent + 1 < gameState.getNumAgents() else 0
                new_depth = depth + 1 if next_agent == 0 else depth
                actions = gameState.getLegalActions(agent)
                return sum(expectimax(next_agent, new_depth, gameState.generateSuccessor(agent, action)) for action in actions) / len(actions)

        # Start from Pac-Man (agent 0) and depth 0
        legalMoves = gameState.getLegalActions(0)
        scores = [expectimax(1, 0, gameState.generateSuccessor(0, action)) for action in legalMoves]
        bestMove = max(scores)
        bestIndices = [index for index, value in enumerate(scores) if value == bestMove]
        chosenIndex = random.choice(bestIndices)  # Choose randomly among the best
        return legalMoves[chosenIndex]


def betterEvaluationFunction(currentGameState):
    from util import manhattanDistance

    if currentGameState.isWin():
        return float('inf')
    elif currentGameState.isLose():
        return -float('inf')

    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    foodList = food.asList()
    minFoodDistance = min(manhattanDistance(pos, foodPos) for foodPos in foodList) if foodList else 1

    # Consider all ghosts and their distances
    ghostDistances = [manhattanDistance(pos, ghost.getPosition()) for ghost in ghostStates]
    minGhostDistance = min(ghostDistances) if ghostDistances else 10

    # Weigh ghost distance taking into account their scared state
    ghostPenalties = sum((2 / (distance + 0.1) if ghostStates[i].scaredTimer == 0 else -10 / (distance + 0.1) for i, distance in enumerate(ghostDistances)))

    # Increased food attraction
    foodAttraction = sum(5 / (manhattanDistance(pos, food) + 1) for food in foodList)

    score = currentGameState.getScore()
    score += foodAttraction - ghostPenalties  # Combine food attraction with ghost penalties

    return score

better=betterEvaluationFunction