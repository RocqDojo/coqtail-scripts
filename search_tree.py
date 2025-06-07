from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable
from typing import Callable, Generic, Hashable, TypeVar, final, override

T = TypeVar("T")  # Node (state) type
E = TypeVar("E")  # Edge (transition) type


@final
class TreeNode(Generic[T, E]):
    def __init__(
        self,
        value: T,
        parent: "TreeNode[T, E] | None" = None,
        parent_edge: E | None = None,
    ) -> None:
        assert (parent is not None and parent_edge is not None) or (
            parent is None and parent_edge is None
        )

        self.value: T = value
        self.parent: TreeNode[T, E] | None = parent
        self.parent_edge: E | None = parent_edge
        self.children: list[TreeNode[T, E]] = []
        self.depth: int = 0 if parent is None else parent.depth + 1

    def add_child(self, value: T, edge_data: E) -> "TreeNode[T, E]":
        child = TreeNode(value, parent=self, parent_edge=edge_data)
        self.children.append(child)
        return child


@final
class CursorTree(Generic[T, E]):
    def __init__(self, root_value: T) -> None:
        self.root: TreeNode[T, E] = TreeNode(root_value)
        self.cursors: dict[str, TreeNode[T, E]] = {"main": self.root}

    def add_cursor(self, name: str, node: TreeNode[T, E] | None = None) -> None:
        assert name not in self.cursors
        self.cursors[name] = node if node else self.root

    def get_cursor(self, name: str = "main") -> TreeNode[T, E] | None:
        return self.cursors.get(name)

    def move_cursor_to(self, name: str, node: TreeNode[T, E]) -> None:
        self.cursors[name] = node

    def add_child_at_cursor(self, name: str, value: T, edge_data: E) -> TreeNode[T, E]:
        cursor = self.get_cursor(name)
        if cursor is None:
            raise ValueError(f"Cursor '{name}' does not exist.")
        return cursor.add_child(value, edge_data)

    def get_edge_path_between(
        self,
        node_a: TreeNode[T, E],
        node_b: TreeNode[T, E],
    ) -> tuple[
        list[tuple[TreeNode[T, E], E]],  # upward  A → LCA   (via parent-edges)
        list[tuple[TreeNode[T, E], E]],  # downward LCA → B (via child-edges)
    ]:
        if node_a is node_b:
            return ([], [])

        # --- collect ancestor chains ----------------------------------
        def chain(n: TreeNode[T, E]) -> list[TreeNode[T, E]]:
            out: list[TreeNode[T, E]] = []
            cur: TreeNode[T, E] | None = n
            while cur:
                out.append(cur)
                cur = cur.parent
            return out

        a_chain, b_chain = chain(node_a), chain(node_b)
        lca = next(n for n in b_chain if n in a_chain)  # lowest common ancestor

        # --- upward part  (node_a  → LCA) -----------------------------
        upward: list[tuple[TreeNode[T, E], E]] = []
        cur = node_a
        while cur is not lca:
            assert (
                cur is not None
                and cur.parent is not None
                and cur.parent_edge is not None
            )
            upward.append((cur, cur.parent_edge))
            cur = cur.parent

        # --- downward part (LCA → node_b) -----------------------------
        downward_nodes: list[TreeNode[T, E]] = []
        cur = node_b
        while cur is not lca:
            assert (
                cur is not None
                and cur.parent is not None
                and cur.parent_edge is not None
            )
            downward_nodes.append(cur)
            cur = cur.parent

        downward_nodes.reverse()  # from LCA’s child downward

        downward: list[tuple[TreeNode[T, E], E]] = [
            (n.parent, n.parent_edge)  # type: ignore[arg-type]
            for n in downward_nodes
        ]

        return (upward, downward)

    def edge_path(
        self,
        from_node: TreeNode[T, E],
        to_node: TreeNode[T, E],
    ) -> list[E]:
        """
        Return the *edge* sequence that carries you from `from_node`
        to `to_node` in **forward order**.
        """

        upward, downward = self.get_edge_path_between(from_node, to_node)

        # 1) upward part is already ordered from from_node → LCA
        # 2) downward part is already ordered from LCA → to_node
        edges: list[E] = [edge for _, edge in upward]
        edges.extend(edge for _, edge in downward)
        return edges


def test_cursor_tree():
    tree = CursorTree[str, int]("root")

    # --- build the tree ----------------------------------------------
    tree.add_cursor("c1")
    tree.add_cursor("c2")

    a = tree.add_child_at_cursor("c1", "A", 1)  # root → A
    b = tree.add_child_at_cursor("c1", "B", 2)  # root → B

    tree.move_cursor_to("c1", a)
    c = tree.add_child_at_cursor("c1", "C", 3)  # A → C
    d = tree.add_child_at_cursor("c1", "D", 4)  # A → D

    tree.move_cursor_to("c2", b)
    e = tree.add_child_at_cursor("c2", "E", 5)  # B → E

    # -----------------------------------------------------------------
    def paths(u: TreeNode[str, int], v: TreeNode[str, int]):
        up, down = tree.get_edge_path_between(u, v)
        up_s = [(n.value, e) for n, e in up]
        down_s = [(n.value, e) for n, e in down]
        return (up_s, down_s)

    # C → D  :  C ↑ A  ↓ D
    assert paths(c, d) == (
        [("C", 3)],  # up
        [("A", 4)],  # down
    )

    # C → E  :  C ↑ A ↑ root  ↓ B ↓ E
    assert paths(c, e) == (
        [("C", 3), ("A", 1)],
        [("root", 2), ("B", 5)],
    )

    # A → D  :  (no up)  ↓ D
    assert paths(a, d) == (
        [],
        [("A", 4)],
    )

    # E → B  :  ↑ B  (no down)
    assert paths(e, b) == (
        [("E", 5)],
        [],
    )

    # C → C  :  empty both ways
    assert paths(c, c) == ([], [])

    print("All tests passed.")


class Env(ABC, Generic[T, E]):
    """Abstract base class describing an external environment."""

    # ------------- navigation ------------------------------------------------
    @abstractmethod
    def apply(self, step: E) -> bool:
        """
        Mutate the environment by taking a step forward.
        Change the external state and return True if the move succeded.
        Otherwise, the external state remains unchanged and False is returned.
        """
        raise NotImplementedError

    @abstractmethod
    def revert(self, step: E) -> bool:
        """
        Undo the previously-taken transition (move *backward*).
        Change the external state and return True if the previous move is reverted.
        Otherwise, the external state remains unchanged and False is returned.
        """
        raise NotImplementedError

    # ------------- inspection ------------------------------------------------
    @abstractmethod
    def current_state(self) -> T:
        """Return a representation of the current state."""
        raise NotImplementedError

    @abstractmethod
    def is_win(self) -> bool:
        """`True` iff the current state is a terminal *winning* position."""
        raise NotImplementedError


# ------------------------------------------------------------------
# move_env uses the two lists separately
# ------------------------------------------------------------------
def move_env(
    tree: CursorTree[T, E],
    env: Env[T, E],
    from_node: TreeNode[T, E],
    to_node: TreeNode[T, E],
) -> None:
    if from_node is to_node:
        return

    upward, downward = tree.get_edge_path_between(from_node, to_node)

    # 1) climb up: revert each edge in order
    for _, edge in upward:
        if not env.revert(edge):
            raise RuntimeError("env.revert() failed while climbing upward")

    # 2) descend: apply each edge in order
    for _, edge in downward:
        if not env.apply(edge):
            raise RuntimeError("env.apply() failed while descending")


MoveGenerator = Callable[[T], Iterable[E]]
"""
A user-supplied callback.

    moves = candidate_moves(state)

should yield *candidate* steps that the algorithm will try in order.
"""


@final
class BFSearch(Generic[T, E]):
    """
    Breadth-first traversal that returns a *shortest* edge list
    from the initial state to a winning state.

    Edge costs are assumed uniform (1 per move).  For weighted graphs
    use Dijkstra or A* instead.
    """

    def __init__(
        self,
        env: Env[T, E],
        move_gen: MoveGenerator[T, E],
        *,
        max_depth: int | None = None,
        avoid_repeats: bool = True,
    ) -> None:
        self.env = env
        self.move_gen = move_gen
        self.max_depth = max_depth
        self.avoid_repeats = avoid_repeats

        root_state = env.current_state()
        self.tree: CursorTree[T, E] = CursorTree(root_state)
        self._visited: set[Hashable] = {root_state}

    # ------------------------------------------------------------------
    def run(self) -> list[E] | None:
        """
        Launch BFS.

        Returns
        -------
        list[E] | None
            Shortest edge-sequence **from the root to a winning state**,
            or `None` if no goal is found (within `max_depth`, if given).
        """

        queue: deque[TreeNode[T, E]] = deque([self.tree.root])
        cursor: TreeNode[T, E] = self.tree.root  # the node where `env` is now

        while queue:
            node = queue.popleft()

            # -- synchronise the external env to this node --------------
            move_env(self.tree, self.env, cursor, node)
            cursor = node

            # -- goal test ----------------------------------------------
            if self.env.is_win():
                return self.tree.edge_path(self.tree.root, node)

            # -- depth limit --------------------------------------------
            if self.max_depth is not None and node.depth >= self.max_depth:
                continue  # do not expand this node any further

            # -- branch expansion --------------------------------------
            for step in self.move_gen(node.value):
                if not self.env.apply(step):  # illegal move
                    continue

                child_state = self.env.current_state()
                if self.avoid_repeats and child_state in self._visited:
                    assert self.env.revert(step), "Cannot revert the last move"
                    continue

                # create child node in tree / frontier
                self._visited.add(child_state)
                child = node.add_child(child_state, step)
                queue.append(child)

                # restore env to `node` so that subsequent moves start
                # from the correct state
                assert self.env.revert(step), "Cannot revert the last move"

        # Exhausted the frontier – no solution
        return None


def test_bfs_search() -> None:
    @final
    class Grid2x2(Env[tuple[int, int], str]):
        """Toy grid to illustrate."""

        def __init__(self):
            self.x, self.y = 0, 0

        @override
        def current_state(self):
            return (self.x, self.y)

        @override
        def is_win(self):
            return (self.x, self.y) == (1, 1)

        @override
        def apply(self, step: str) -> bool:
            nx, ny = self.x, self.y
            if step == "U":
                ny -= 1
            elif step == "D":
                ny += 1
            elif step == "L":
                nx -= 1
            elif step == "R":
                nx += 1
            else:
                return False
            if 0 <= nx <= 1 and 0 <= ny <= 1:
                self.x, self.y = nx, ny
                return True
            return False

        @override
        def revert(self, step: str) -> bool:
            opp = {"U": "D", "D": "U", "L": "R", "R": "L"}.get(step)
            return self.apply(opp) if opp else False

    def four_dirs(_: tuple[int, int]) -> Iterable[str]:
        yield from ("U", "D", "L", "R")

    env = Grid2x2()
    solver = BFSearch(env, four_dirs)
    assert solver.run() == ["D", "R"]


if __name__ == "__main__":
    test_cursor_tree()
    test_bfs_search()
