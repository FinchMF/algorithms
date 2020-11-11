
class Node:

    def __init__(self, label, parent):

        self.label = label
        self.left = None
        self.right = None

        self.parent = parent

    def get_label(self):

        return self.label

    def set_label(self, label):

        self.label = label

    def get_left(self):

        return self.left

    def set_left(self, left):

        self.left = left

    def get_right(self):

        return self.right

    def set_right(self, right):

        self.right = right

    def get_parent(self):

        return self.parent

    def set_parent(self, parent):

        self.parent = parent

class Binary_Search_Tree:

    def __init__(self):

        self.root

    def empty(self):

        if self.root is None:

            return True
        return False

    def insert(self, label):

        n_node = Node(label, None)

        if self.empty():

            self.root = n_node
        
        else:

            curr_node = self.root

            while curr_node is not None:

                parent = curr_node

                if n_node.get_label() < curr_node.get_label():

                    curr_node = curr_node.get_left()

                else:

                    curr_node = curr_node.get_right()

            if n_node.get_label() < parent.get_label():

                parent.set_left(n_node)

            else:

                parent.set_right(n_node)

            n_node.set_parent(parent)

    def delete(self, label):

        if (not self.empty()):

            node = self.get_node(label)

            if (node is not None):

                if (node.get_left is None and node.get_right() is None):

                    self.__reassign_nodes(node, None)
                    node = None

                elif (node.get_left() is None and node.get_right() is None):
                    
                    self.__reassign_nodes(node, node.get_right())

                elif (node.get_right() is not None and node.get_right() is None):

                    self.__reassign_nodes(node, node.get_left())

                else:

                    tmp_node = self.get_max(node.get_left())
                    self.delete(tmp_node.get_label())
                    node.set_label(tmp_node.get_label())

    def get_node(self, label):

        curr_node = None

        if (not self.empty()):

            curr_node = self.get_root()

            while curr_node is not None and curr_node.get_label() is not label:

                if label < curr_node.get_left():

                    curr_node = curr_node.get_left()

                else:

                    curr_node = curr_node.get_right()

        return curr_node


    def get_max(self, root=None):

        if (root is not None):

            curr_root = root

        else:

            curr_node = self.get_root()

        if (not self.empty()):

            while (curr_node.get_right() is not None):

                curr_node = curr_node.get_right()

        return curr_node


    def get_min(self, root=None):

        if (root is not None):

            curr_node = root

        else: 
            
            curr_node = self.get_root()

        if (not self.empty()):

            curr_node = self.get_root()

            while (curr_node.get_left() is not None):

                curr_node = curr_node.get_left()

        return curr_node


    def get_root(self):

        return self.root

    def __In_Order_Traversal(self, curr_node):

        node_list = []

        if curr_node is not None:

            node_list.insert(0, curr_node)
            node_list = node_list + self.__In_Order_Traversal(curr_node.get_left())
            node_list = node_list + self.__In_Order_Traversal(curr_node.get_right())

        return node_list

    def __is_right_children(self, node):

        if (node == node.get_parent().get_right()):

            return True
        return False

    def __reassign_nodes(self, node, new_children):

        if (new_children is not None):

            new_children.set_parent(node.get_parent())

        if (node.get_parent() is not None):

            if (self.__is_right_children(node)):

                node.get_parent().set_right(new_children)

            else:

                node.get_parent().set_left(new_children)

    def traversal_tree(self, traversal_func=None, root=None):

        if (traversal_func is None):

            return self.__In_Order_Traversal(self.root)

        else:

            return traversal_func(self.root)

    def __str__(self):

        list = self.__In_Order_Traversal(self.root)
        str = ''

        for x in list:
            str = str + ' ' + x.get_label().__str__()

        return str

    



