
from utils import *


row_units = [cross(r, cols) for r in rows]
column_units = [cross(rows, c) for c in cols]
square_units = [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')]

diagonal_units1=[]

for r in range(len(rows)):
    diagonal_units1.append(rows[r]+str(r+1))

diagonal_units2=[]
cont=9
for r in range(len(rows)):
    diagonal_units2.append(rows[r]+str(cont))
    cont=cont-1

diagonal_units=[diagonal_units1,diagonal_units2]

unitlist = row_units + column_units + square_units

# TODO: Update the unit list to add the new diagonal units
unitlist = unitlist+diagonal_units


# Must be called after all units (including diagonals) are added to the unitlist
units = extract_units(unitlist, boxes)
peers = extract_peers(units, boxes)


def naked_twins(values):
    """Eliminate values using the naked twins strategy.

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict
        The values dictionary with the naked twins eliminated from peers

    Notes
    -----
    Your solution can either process all pairs of naked twins from the input once,
    or it can continue processing pairs of naked twins until there are no such
    pairs remaining -- the project assistant test suite will accept either
    convention. However, it will not accept code that does not process all pairs
    of naked twins from the original input. (For example, if you start processing
    pairs of twins and eliminate another pair of twins before the second pair
    is processed then your code will fail the PA test suite.)

    The first convention is preferred for consistency with the other strategies,
    and because it is simpler (since the reduce_puzzle function already calls this
    strategy repeatedly).
    """
    # TODO: Implement this function!
    for unit in unitlist:
        box_list=[box for box in unit if len(values[box])==2]
        values_list=[values[box] for box in box_list]
        for el in box_list:
            #also checks if boxes with numbers ex interchanged:(12 and 21) are in the same unit, and considers this possibility
            x=values[el][1]+values[el][0]
            cond=(x in values_list)and(values[el] in values_list)
            if values_list.count(values[el])==2 or cond:
                digit=values[el]
                values_list.remove(digit)
                box_list.remove(el)
                for peer in unit:
                    find=False
                    if values[peer]!=digit and digit in values[peer]:
                        values[peer] = values[peer].replace(digit,'')
                        find=True
                    if values[peer]!=x and cond:
                        values[peer] = values[peer].replace(x,'')
                        find=True
                    if values[peer]!=digit and digit[0] in values[peer]:
                        values[peer] = values[peer].replace(digit[0],'')
                        find=True
                    if values[peer]!=digit and digit[1] in values[peer]:
                        values[peer] = values[peer].replace(digit[1],'')
                        find=True
                    if len(values[peer])==2 and find:
                        box_list.append(peer)
                        values_list.append(values[peer])
                    
        
          
    return values        
    

def eliminate(values):
    """Apply the eliminate strategy to a Sudoku puzzle

    The eliminate strategy says that if a box has a value assigned, then none
    of the peers of that box can have the same value.

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict
        The values dictionary with the assigned values eliminated from peers
    """
    # TODO: Copy your code from the classroom to complete this function
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    for box in solved_values:
        digit = values[box]
        for peer in peers[box]:
            values[peer] = values[peer].replace(digit,'')
    return values


def only_choice(values):
    """Apply the only choice strategy to a Sudoku puzzle

    The only choice strategy says that if only one box in a unit allows a certain
    digit, then that box must be assigned that digit.

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict
        The values dictionary with all single-valued boxes assigned

    Notes
    -----
    You should be able to complete this function by copying your code from the classroom
    """
    # TODO: Copy your code from the classroom to complete this function
    for unit in unitlist:
        for digit in '123456789':
            dplaces = [box for box in unit if digit in values[box]]
            if len(dplaces) == 1:
                values[dplaces[0]] = digit
    return values


def reduce_puzzle(values):
    """Reduce a Sudoku puzzle by repeatedly applying all constraint strategies

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict or False
        The values dictionary after continued application of the constraint strategies
        no longer produces any changes, or False if the puzzle is unsolvable 
    """
    # TODO: Copy your code from the classroom and modify it to complete this function
    solved_values = [box for box in values.keys() if len(values[box]) == 1]
    stalled = False
    while not stalled:
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])
        values = eliminate(values)
        values = only_choice(values)
        values=naked_twins(values)
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        stalled = solved_values_before == solved_values_after
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values


def search(values):
    """Apply depth first search to solve Sudoku puzzles in order to solve puzzles
    that cannot be solved by repeated reduction alone.

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict or False
        The values dictionary with all boxes assigned or False

    Notes
    -----
    You should be able to complete this function by copying your code from the classroom
    and extending it to call the naked twins strategy.
    """
    # TODO: Copy your code from the classroom to complete this function
    "Using depth-first search and propagation, create a search tree and solve the sudoku."
    # First, reduce the puzzle using the previous function
    values=reduce_puzzle(values)
    if values is False:
        return False
    # Choose one of the unfilled squares with the fewest possibilities
    menor=9
    valor=''
    for i in values:
        if len(values[i])<menor and len(values[i])>1:
            menor=len(values[i])
            valor=i
    
    width = max(len(values[s]) for s in values)
    
    if width==1:
        return values
    else:
        
        # Now use recursion to solve each one of the resulting sudokus, and if one returns a value (not False), return that answer!
        for i in values[valor]:
            #print(i,'-',values[valor],'-',valor)
            new_sudoku=values.copy()
            new_sudoku[valor]=i
            #display(new_sudoku)
            resultado=search(new_sudoku)
            if resultado:
                return resultado
           
            


def solve(grid):
    """Find the solution to a Sudoku puzzle using search and constraint propagation

    Parameters
    ----------
    grid(string)
        a string representing a sudoku grid.
        
        Ex. '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'

    Returns
    -------
    dict or False
        The dictionary representation of the final sudoku grid or False if no solution exists.
    """
    values = grid2values(grid)
    values = search(values)
    return values


if __name__ == "__main__":
    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    display(grid2values(diag_sudoku_grid))
    result = solve(diag_sudoku_grid)
    display(result)

    try:
        import PySudoku
        PySudoku.play(grid2values(diag_sudoku_grid), result, history)

    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
