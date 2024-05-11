import sys
import pygraphviz as pgv

def parse_tree_text(tree_text):
    lines = tree_text.split('\n')
    root = None
    current_level = -1
    stack = []

    for line in lines:
        if line.strip() == '':
            continue

        level = line.count('|--')
        label = line.strip().replace('|', '').replace('--', '').strip()

        if level == 0:
            root = Node(label)
            stack.append(root)
            current_level = 0
        elif level > current_level:
            parent = stack[-1]
            node = Node(label, parent)
            parent.add_child(node)
            stack.append(node)
            current_level = level
        elif level == current_level:
            stack.pop()
            parent = stack[-1]
            node = Node(label, parent)
            parent.add_child(node)
            stack.append(node)
        else:
            while level < current_level:
                stack.pop()
                current_level -= 1
            parent = stack[-1]
            node = Node(label, parent)
            parent.add_child(node)
            stack.append(node)
            current_level = level

    return root


class Node:
    def __init__(self, label, parent=None):
        self.label = label
        self.parent = parent
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def to_graph(self, graph):
        for child in self.children:
            graph.add_edge(self.label, child.label)
            child.to_graph(graph)

    def print_tree(self, prefix=''):
        print(prefix + self.label)
        for child in self.children:
            child.print_tree(prefix + '|   ')


def generate_tree_visualization(tree_text, output_file):
    root = parse_tree_text(tree_text)

    graph = pgv.AGraph(directed=True)
    root.to_graph(graph)

    graph.layout(prog='dot')
    graph.draw(output_file)


# 从命令行参数读取 tree 命令生成的文本和输出文件路径
if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('请提供 tree 命令生成的文本文件和输出 PNG 图片文件路径作为参数！')
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(input_file, 'r') as file:
        tree_text = file.read()

    generate_tree_visualization(tree_text, output_file)
