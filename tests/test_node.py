from src.mcts.node import Node
from src.othello.board import Board
from src.neural_net.model import dummy_model_predict
from src.utils.index_to_coordinates import index_to_coordinates
import numpy as np

def test_node_creation():
    # Create a node with None parameters (using default values)
    node = Node(prior=None)
    
    # Assert that the node was successfully created and is not None
    assert node is not None, "Node creation failed. The node is None."
    
    # Assert that the prior is None, since we passed None for it during initialization
    assert node.prior is None, f"Expected prior to be None, but got {node.prior}"
    
    # Assert that the board is an instance of the Board class
    # This ensures that the node correctly initializes a Board object
    assert isinstance(node.board, Board), f"Expected board to be of type Board, but got {type(node.board)}"
    
    # Assert that children are initialized as an empty dictionary
    # This ensures that when a node is created, it has no children initially
    assert node.children == {}, f"Expected children to be an empty dictionary, but got {node.children}"
    
    # Assert that the value is initialized to 0
    # The node's value should default to 0 when it is created
    assert node.value == 0, f"Expected value to be 0, but got {node.value}"
    
    # Assert that visits are initialized to 0
    # The node's visit count should start at 0, as no visits have occurred yet
    assert node.visits == 0, f"Expected visits to be 0, but got {node.visits}"

    # Print confirmation that the node creation test has passed
    print("Node creation test passed!")

def test_expand_valid_moves():
    # Create a node with a simple (empty) board
    node = Node()
    
    # Assert that no children exist initially
    # When a node is first created, it should have no child nodes
    assert len(node.children) == 0, f"Expected no children, but found {len(node.children)}"
    
    # Define action probabilities with non-zero values for some valid moves
    # We use the dummy model's prediction function to simulate action probabilities
    action_probs, value = dummy_model_predict(node.board.board)
    
    # Expand the node with the action probabilities
    # This simulates the process of expanding the node based on predicted action probabilities
    node.expand(action_probs)
    
    # Assert that the node has children after expansion
    # The exact number of children depends on the board state and valid moves.
    # For this test, we expect 4 valid children (e.g., valid moves based on action probabilities).
    assert len(node.children) in [4], f"Expected [4] children, but found {len(node.children)}"
    
    # Print confirmation that the expand valid moves test has passed
    print("test_expand_valid_moves passed!")

def test_expand_with_invalid_move():
    # Create a node with a simple board
    node = Node()
    
    # Apply a move to occupy the (0, 0) position
    # This simulates the action of a player making a move, which occupies a spot on the board
    node.board.apply_move(0, 0)
    
    # Define action probabilities with non-zero values for some moves
    # Use the dummy model to get action probabilities for the current board state
    action_probs, value = dummy_model_predict(node.board.board)
    
    # Expand the node with the action probabilities
    # This expands the node based on the valid moves determined by the action probabilities
    node.expand(action_probs)
    
    # Assert that (0, 0) is not in the children since it was occupied
    # The move (0, 0) was already taken, so it should not be a valid child node
    assert (0, 0) not in node.children, f"Expected no child at (0, 0), but found it"
    
    # Assert that valid moves are present in the children
    # The expected valid move indices (e.g., 19, 26, 37, 44) are derived from the action probabilities
    # These values correspond to valid moves that the model has predicted for the current board
    assert [19, 26, 37, 44] == list(node.children.keys()), f"Expected a child at {(19, 26, 37, 44)}, but it found {list(node.children.keys())}"
   
    # Print confirmation that the expand with invalid move test has passed
    print("test_expand_with_invalid_move passed!")

def test_expand_with_no_valid_moves():
    # Create a node with a full board (no valid moves)
    node = Node()
    
   
    
    # Define action probabilities with all values set to 0
    # Since there are no valid moves left, the action probabilities are all zero
    action_probs = [0] * 64
    
    # Expand the node with the action probabilities
    # Since there are no valid moves, the expand function should not create any child nodes
    node.expand(action_probs)
    
    # Assert that no children are created since there are no valid moves
    # As the board is full, there should be no possible valid moves to expand to
    assert len(node.children) == 0, f"Expected no children, but found {len(node.children)}"
    
    # Print confirmation that the expand with no valid moves test has passed
    print("test_expand_with_no_valid_moves passed!")


def test_expand_children_independent_boards():
    # Create a node with an initial board state
    node = Node()
    
    # Define action probabilities for some valid moves
    # Assuming valid moves are represented as coordinates (in this case, let's say moves at (0,0), (0,1), (1,0), (1,1))
    action_prob, value = dummy_model_predict(node.board.board) # Non-zero probabilities for 4 valid moves
    
    # Expand the node with the action probabilities
    node.expand(action_prob)
    
    # Assert that there are 4 children, each corresponding to a valid move
    assert len(node.children) == 4, f"Expected 4 children, but found {len(node.children)}"
    
    # Track the board states of the children to check if they are independent
    child_boards = []
    child_moves = []
    
    # Iterate over the children and verify their board states and moves
    for action, child in node.children.items():
        # Ensure each child has a different board state
        child_boards.append(child.board.board)  # Get the board of the child
        child_moves.append(index_to_coordinates(action))  # Get the move made by the child
        
        # Ensure that no two children share the same board state (they must have independent boards)
        assert all(
            not np.array_equal(child.board.board, other_board) for other_board in child_boards[:-1]
        ), f"Child with action {action} shares the same board state as a previous child."

    # Check that all moves are distinct
    assert len(child_moves) == len(set(child_moves)), "Expected all moves to be distinct, but some moves are the same."

    # Check that the children have correctly applied different moves
    for action, child in node.children.items():
        x, y = index_to_coordinates(action)  # Convert action to coordinates
        assert (x, y) != (0, 0), f"Expected child with action {action} to play a move other than (0, 0)."

    print("test_expand_children_independent_boards passed!")



def test_select_child():
    # Create a node with a simple board
    node = Node()
    node.visits = 1
    
    # Add some children to the node (manually or using a method like expand)
    # For simplicity, let's assume ucb_score is a known function that gives higher scores to some children
    node.children = {
        0: Node(prior=0.3,  board=Board()),  # Add child with low prior value
        1: Node(prior=0.5, board=Board()),  # Add child with a medium prior value
        2: Node(prior=0.8, board=Board())   # Add child with a high prior value
    }
    for n in node.children.values():
        n.visits = 1
    
    # Select the child with the highest UCB score
    action, selected_child = node.select_child()
    
    # Assert that the child with the highest prior is selected (since UCB score depends on it)
    # Assuming `ucb_score` is calculated based on prior, the child with action 2 should be selected
    assert action == 2, f"Expected action 2 (highest UCB score), but got {action}"
    
    # Assert that the selected child has the highest prior value (0.8)
    assert selected_child.prior == 0.8, f"Expected prior of 0.8, but got {selected_child.prior}"
    
    # Print confirmation that the select child test has passed
    print("test_select_child passed!")

def test_is_expanded():
    # Create a node with a simple board
    node = Node()
    
    # Assert that the node has no children initially, so it is not expanded
    assert not node.is_expanded(), "Expected is_expanded to return False, but it returned True"
    
    # Add children manually or by expanding the node
    node.children = {0: Node(prior=0.3, board=Board())}  # Adding a single child
    
    # Assert that after adding a child, the node is expanded
    assert node.is_expanded(), "Expected is_expanded to return True after expansion"
    
    # Print confirmation that the is_expanded test has passed
    print("test_is_expanded passed!")

def test_is_terminal_state():
    # Create a node with a simple board
    node = Node()
    
    # Mock the Board's `is_terminal_state()` method to return False (game not over)
    node.board.is_terminal_state = lambda: False  # Overriding the method for testing
    
    # Assert that the node's board is not in a terminal state
    assert not node.is_terminal_state(), "Expected is_terminal_state to return False, but it returned True"
    
    # Mock the Board's `is_terminal_state()` method to return True (game over)
    node.board.is_terminal_state = lambda: True  # Overriding the method for testing
    
    # Assert that the node's board is in a terminal state
    assert node.is_terminal_state(), "Expected is_terminal_state to return True, but it returned False"
    
    # Print confirmation that the is_terminal_state test has passed
    print("test_is_terminal_state passed!")
