from mpi4py import MPI
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import networkx as nx
import json
from datetime import datetime
import numpy as np

class Place:
    def __init__(self, name, tokens=0, pos=(0, 0)):
        self.name = name
        self.tokens = tokens
        self.pos = pos

class Transition:
    def __init__(self, name, inputs, outputs, pos=(0, 0)):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.pos = pos

    def is_enabled(self, marking):
        return all(marking[place.name] > 0 for place in self.inputs)

    def fire(self, marking):
        if not self.is_enabled(marking):
            return None
        new_marking = marking.copy()
        for place in self.inputs:
            new_marking[place.name] -= 1
        for place in self.outputs:
            new_marking[place.name] += 1
        return new_marking

class MarkingGraph:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.animation = None
        self.current_frame = 0

    def add_node(self, marking):
        if marking not in self.nodes:
            self.nodes.append(marking)

    def add_edge(self, source, transition, target):
        self.edges.append((source, transition, target))

class ModernPetriNetGUI:
    def __init__(self, is_master=True):
        self.is_master = is_master
        if is_master:
            self.root = tk.Tk()
            self.root.title("Modern Petri Net Designer")
            self.root.geometry("1200x800")
            self.setup_styles()
            self.setup_gui()
        self.places = []
        self.transitions = []
        self.marking_graph = MarkingGraph()
        self.animation = None
        self.transaction_history = []
        
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', padding=6, relief="flat", background="#2196F3")
        style.configure('TLabel', padding=5)
        style.configure('TFrame', background="#f5f5f5")
        
    def setup_gui(self):
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel for controls with scrollbar
        left_panel_frame = ttk.Frame(main_container)
        left_panel_frame.pack(side=tk.LEFT, fill=tk.Y)

        left_panel_canvas = tk.Canvas(left_panel_frame)
        left_panel_scrollbar = ttk.Scrollbar(
            left_panel_frame, orient=tk.VERTICAL, command=left_panel_canvas.yview
        )
        left_panel_scrollable_frame = ttk.Frame(left_panel_canvas)

        # Configure scrollable frame for left panel
        left_panel_scrollable_frame.bind(
            "<Configure>",
            lambda e: left_panel_canvas.configure(
                scrollregion=left_panel_canvas.bbox("all"),
                width=left_panel_scrollable_frame.winfo_reqwidth() 
            )
        )

        left_panel_canvas.create_window((0, 0), window=left_panel_scrollable_frame, anchor="nw")
        left_panel_canvas.configure(yscrollcommand=left_panel_scrollbar.set)

        left_panel_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        left_panel_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Place configuration
        place_frame = ttk.LabelFrame(left_panel_scrollable_frame, text="Add Place")
        place_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(place_frame, text="Name:").pack(padx=5)
        self.place_name = ttk.Entry(place_frame)
        self.place_name.pack(fill=tk.X, padx=5)

        ttk.Label(place_frame, text="Tokens:").pack(padx=5)
        self.place_tokens = ttk.Entry(place_frame)
        self.place_tokens.pack(fill=tk.X, padx=5)

        ttk.Button(place_frame, text="Add Place", command=self.add_place).pack(pady=5, padx=5)

        # Scrollable Places Listbox
        places_listbox_frame = ttk.Frame(place_frame)
        places_listbox_frame.pack(fill=tk.X, padx=5)

        places_scrollbar = ttk.Scrollbar(places_listbox_frame, orient=tk.VERTICAL)
        self.places_listbox = tk.Listbox(places_listbox_frame, height=4, yscrollcommand=places_scrollbar.set)
        self.places_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        places_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        places_scrollbar.config(command=self.places_listbox.yview)

        # Transition configuration
        trans_frame = ttk.LabelFrame(left_panel_scrollable_frame, text="Add Transition")
        trans_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(trans_frame, text="Name:").pack(padx=5)
        self.trans_name = ttk.Entry(trans_frame)
        self.trans_name.pack(fill=tk.X, padx=5)

        # Input Places Listbox with Scrollbar
        ttk.Label(trans_frame, text="Input Places:").pack(padx=5)
        input_listbox_frame = ttk.Frame(trans_frame)
        input_listbox_frame.pack(fill=tk.X, padx=5, pady=5)

        input_places_scrollbar = ttk.Scrollbar(input_listbox_frame, orient=tk.VERTICAL)
        self.trans_input_listbox = tk.Listbox(
            input_listbox_frame, selectmode=tk.MULTIPLE, height=4, exportselection=False, yscrollcommand=input_places_scrollbar.set
        )
        self.trans_input_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        input_places_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        input_places_scrollbar.config(command=self.trans_input_listbox.yview)

        # Output Places Listbox with Scrollbar
        ttk.Label(trans_frame, text="Output Places:").pack(padx=5)
        output_listbox_frame = ttk.Frame(trans_frame)
        output_listbox_frame.pack(fill=tk.X, padx=5, pady=5)

        output_places_scrollbar = ttk.Scrollbar(output_listbox_frame, orient=tk.VERTICAL)
        self.trans_output_listbox = tk.Listbox(
            output_listbox_frame, selectmode=tk.MULTIPLE, height=4, exportselection=False, yscrollcommand=output_places_scrollbar.set
        )
        self.trans_output_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        output_places_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        output_places_scrollbar.config(command=self.trans_output_listbox.yview)

        ttk.Button(trans_frame, text="Add Transition", command=self.add_transition).pack(pady=5, padx=5)

        # Scrollable Transitions Listbox
        transitions_listbox_frame = ttk.Frame(trans_frame)
        transitions_listbox_frame.pack(fill=tk.X, padx=5, pady=5)

        transitions_scrollbar = ttk.Scrollbar(transitions_listbox_frame, orient=tk.VERTICAL)
        self.transitions_listbox = tk.Listbox(transitions_listbox_frame, height=4, yscrollcommand=transitions_scrollbar.set)
        self.transitions_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        transitions_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        transitions_scrollbar.config(command=self.transitions_listbox.yview)

        # Transaction history
        history_frame = ttk.LabelFrame(left_panel_scrollable_frame, text="Transaction History")
        history_frame.pack(fill=tk.X, pady=(0, 10))

        history_scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL)
        self.history_listbox = tk.Listbox(history_frame, height=4, yscrollcommand=history_scrollbar.set)
        self.history_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        history_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        history_scrollbar.config(command=self.history_listbox.yview)

        # Right panel for visualization
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Canvas for Petri net visualization
        self.fig_petri, self.ax_petri = plt.subplots(figsize=(6, 4))
        self.canvas_petri = FigureCanvasTkAgg(self.fig_petri, right_panel)
        self.canvas_petri.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Control buttons
        control_frame = ttk.Frame(left_panel_scrollable_frame)
        control_frame.pack(fill=tk.X, pady=10)

        ttk.Button(control_frame, text="Load Example", command=self.load_example).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Explore Petri Net", command=self.explore_and_exit).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Replay Animation", command=self.replay_animation).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Clear All", command=self.clear_all).pack(side=tk.LEFT, padx=5)

    def add_place(self):
        if not self.is_master:
            return

        name = self.place_name.get().strip()
        tokens = self.place_tokens.get().strip()

        if not name:
            messagebox.showerror("Error", "Place name cannot be empty")
            return
        if not tokens.isdigit():
            messagebox.showerror("Error", "Tokens must be a positive number")
            return
        if any(p.name == name for p in self.places):
            messagebox.showerror("Error", f"Place '{name}' already exists")
            return

        place = Place(name, int(tokens))
        self.places.append(place)
        self.places_listbox.insert(tk.END, f"{name}: {tokens} tokens")

        # Update listboxes for transitions
        self.trans_input_listbox.insert(tk.END, name)
        self.trans_output_listbox.insert(tk.END, name)

        self.place_name.delete(0, tk.END)
        self.place_tokens.delete(0, tk.END)

        self.update_petri_visualization()
        self.add_to_history(f"Added place: {name}")
    def add_transition(self):
        if not self.is_master:
            return

        name = self.trans_name.get().strip()

        if not name:
            messagebox.showerror("Error", "Transition name cannot be empty")
            return
        if any(t.name == name for t in self.transitions):
            messagebox.showerror("Error", f"Transition '{name}' already exists")
            return

        # Get selected input and output places
        selected_input_indices = self.trans_input_listbox.curselection()
        selected_output_indices = self.trans_output_listbox.curselection()

        if not selected_input_indices or not selected_output_indices:
            messagebox.showerror("Error", "Please select at least one input and one output place")
            return

        # Map indices to place names
        selected_inputs = [self.trans_input_listbox.get(i) for i in selected_input_indices]
        selected_outputs = [self.trans_output_listbox.get(i) for i in selected_output_indices]

        # Find the corresponding place objects
        input_places = [p for p in self.places if p.name in selected_inputs]
        output_places = [p for p in self.places if p.name in selected_outputs]

        # Create and add the transition
        transition = Transition(name, input_places, output_places)
        self.transitions.append(transition)
        self.transitions_listbox.insert(
            tk.END, f"{name}: {', '.join(selected_inputs)} -> {', '.join(selected_outputs)}"
        )

        # Clear fields after adding
        self.trans_name.delete(0, tk.END)
        self.trans_input_listbox.selection_clear(0, tk.END)
        self.trans_output_listbox.selection_clear(0, tk.END)

        self.update_petri_visualization()
        self.add_to_history(f"Added transition: {name}")

    def add_to_history(self, action):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.transaction_history.append(f"{timestamp} - {action}")
        self.history_listbox.insert(0, f"{timestamp} - {action}")

    def update_petri_visualization(self):
        self.ax_petri.clear()
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add places
        pos = {}
        for i, place in enumerate(self.places):
            G.add_node(place.name, type='place', tokens=place.tokens)
            pos[place.name] = (i * 2, 0)
            
        # Add transitions
        for i, transition in enumerate(self.transitions):
            trans_name = f"T_{transition.name}"
            G.add_node(trans_name, type='transition')
            pos[trans_name] = (i * 2 + 1, 1)
            
            # Add edges
            for place in transition.inputs:
                G.add_edge(place.name, trans_name)
            for place in transition.outputs:
                G.add_edge(trans_name, place.name)

        # Draw the graph
        nx.draw_networkx_nodes(G, pos, 
                             nodelist=[n for n, d in G.nodes(data=True) if d['type'] == 'place'],
                             node_color='lightblue',
                             node_size=1000)
        
        nx.draw_networkx_nodes(G, pos,
                             nodelist=[n for n, d in G.nodes(data=True) if d['type'] == 'transition'],
                             node_color='lightgreen',
                             node_shape='s',
                             node_size=800)

        # Draw edges with arrows
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos)
        
        # Draw tokens as small circles inside places
        for node, attr in G.nodes(data=True):
            if attr.get('type') == 'place':
                tokens = attr['tokens']
                if tokens > 0:
                    center_x, center_y = pos[node]
                    
                    # Calculate positions for tokens in a circular arrangement
                    for i in range(tokens):
                        angle = 2 * np.pi * i / max(tokens, 1)
                        if tokens == 1:
                            token_x, token_y = center_x, center_y
                        else:
                            token_x = center_x + 0.15 * 0.5 * np.cos(angle)
                            token_y = center_y + 0.15 * 0.5 * np.sin(angle)
                        self.ax_petri.plot(token_x, token_y, 'ko', markersize=8)

        self.ax_petri.set_title("Petri Net Visualization")
        self.canvas_petri.draw()

    def draw_static_elements(self):
        """Draw the static elements of the Petri net"""
        # Draw basic Petri net structure
        G = nx.DiGraph()
        
        # Add places and transitions
        for place in self.places:
            G.add_node(place.name, type='place')
        for trans in self.transitions:
            trans_name = f"T_{trans.name}"
            G.add_node(trans_name, type='transition')
            for p in trans.inputs:
                G.add_edge(p.name, trans_name)
            for p in trans.outputs:
                G.add_edge(trans_name, p.name)

        # Draw places
        nx.draw_networkx_nodes(G, self.pos,
                             nodelist=[n for n, d in G.nodes(data=True) if d['type'] == 'place'],
                             node_color='lightblue',
                             node_size=1000)
        
        # Draw transitions
        nx.draw_networkx_nodes(G, self.pos,
                             nodelist=[n for n, d in G.nodes(data=True) if d['type'] == 'transition'],
                             node_color='lightgreen',
                             node_shape='s',
                             node_size=800)
        
        # Draw edges and labels
        nx.draw_networkx_edges(G, self.pos, edge_color='gray', arrows=True, arrowsize=20)
        nx.draw_networkx_labels(G, self.pos)

    def animate_marking(self, marking_graph):
        """Animate the marking graph with token movement"""
        if not marking_graph.edges:
            messagebox.showinfo("No Animation", "No transitions to animate. Add more places and transitions.")
            return

        print("Starting animation...")
        print(f"Edges: {marking_graph.edges}")

        # Store positions for animation
        self.pos = {}
        for i, place in enumerate(self.places):
            self.pos[place.name] = (i * 2, 0)
        for i, transition in enumerate(self.transitions):
            self.pos[f"T_{transition.name}"] = (i * 2 + 1, 1)

        # Create a list to track token movements
        token_movements = []
        self.current_marking = marking_graph.nodes[0].copy()  # Start with initial marking
        # Check for cycles in the marking graph
        def has_cycle(edges):
            G = nx.DiGraph()
            for source, _, target in edges:
                G.add_edge(str(source), str(target))
            try:
                return len(list(nx.simple_cycles(G))) > 0
            except:
                return False
        
        is_cyclic = has_cycle(marking_graph.edges)
        print(f"Path contains cycles: {is_cyclic}")
        
        # Pre-calculate all token movements
        processed_edges = set()
        current_marking = None

        def marking_to_hashable(marking):
            """Convert a marking dictionary to a hashable format"""
            return frozenset(marking.items())
        
        def add_movement(edge):
            source_marking, transition_name, target_marking = edge
            transition = next((t for t in self.transitions if t.name == transition_name), None)
            if transition:
                for input_place in transition.inputs:
                    if source_marking[input_place.name] > 0:
                        src_pos = self.pos[input_place.name]
                        trans_pos = self.pos[f"T_{transition.name}"]
                        for output_place in transition.outputs:
                            dst_pos = self.pos[output_place.name]
                            token_movements.append({
                                'path': [src_pos, trans_pos, dst_pos],
                                'source': input_place.name,
                                'transition': transition.name,
                                'target': output_place.name,
                                'source_marking': source_marking.copy(),
                                'target_marking': target_marking.copy()
                            })

        # First, add all edges to get the basic movements
        for edge in marking_graph.edges:
            add_movement(edge)
            # Convert markings to hashable format before adding to processed_edges
            edge_key = (marking_to_hashable(edge[0]), edge[1], marking_to_hashable(edge[2]))
            processed_edges.add(edge_key)
            if current_marking is None:
                current_marking = edge[0].copy()

        # If cyclic, ensure we capture the complete cycle
        if is_cyclic:
            # Find all cycles in the graph
            G = nx.DiGraph()
            for edge in marking_graph.edges:
                source, trans, target = edge
                G.add_edge(str(source), str(target), transition=trans)
            
            try:
                cycles = list(nx.simple_cycles(G))
                for cycle in cycles:
                    # Convert cycle nodes back to markings and add movements
                    for i in range(len(cycle)):
                        current = eval(cycle[i])
                        next_idx = (i + 1) % len(cycle)
                        next_marking = eval(cycle[next_idx])
                        
                        # Find the transition that connects these markings
                        for edge in marking_graph.edges:
                            edge_key = (marking_to_hashable(edge[0]), edge[1], marking_to_hashable(edge[2]))
                            if (str(edge[0]) == cycle[i] and 
                                str(edge[2]) == cycle[next_idx] and
                                edge_key not in processed_edges):
                                add_movement(edge)
                                processed_edges.add(edge_key)
            except Exception as e:
                print(f"Error processing cycles: {str(e)}")

        print(f"Created {len(token_movements)} token movements")
        
        def init():
            self.ax_petri.clear()
            self.draw_static_elements()
            return []

        def update(frame):
            self.ax_petri.clear()
            self.draw_static_elements()  # Redraw the static structure of the Petri net

            # Handle cyclic animations by wrapping the movement_idx
            if is_cyclic:
                movement_idx = frame // 20 % len(token_movements)
            else:
                movement_idx = frame // 20
            
            sub_frame = frame % 20

            # Draw static tokens
            for place in self.places:
                tokens = self.current_marking[place.name]
                if movement_idx < len(token_movements) and place.name == token_movements[movement_idx]['source']:
                    tokens -= 1

                if tokens > 0:
                    center_x, center_y = self.pos[place.name]
                    for i in range(tokens):
                        angle = 2 * np.pi * i / max(tokens, 1)
                        token_x = center_x + 0.15 * 0.5 * np.cos(angle)
                        token_y = center_y + 0.15 * 0.5 * np.sin(angle)
                        self.ax_petri.plot(token_x, token_y, 'ko', markersize=8)

            # Handle moving token
            if movement_idx < len(token_movements):
                movement = token_movements[movement_idx]
                path = movement['path']

                # Calculate position with easing
                if sub_frame < 10:
                    progress = sub_frame / 10
                    start_pos = path[0]
                    end_pos = path[1]
                else:
                    progress = (sub_frame - 10) / 10
                    start_pos = path[1]
                    end_pos = path[2]

                # Snap to endpoint when very close
                if (sub_frame < 10 and sub_frame > 8) or (sub_frame > 18):
                    current_x, current_y = end_pos
                else:
                    current_x = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
                    current_y = start_pos[1] + (end_pos[1] - start_pos[1]) * progress

                # Draw the moving token
                self.ax_petri.plot(current_x, current_y, 'ko', markersize=8)

                # Update marking at the end of movement
                if sub_frame == 19:
                    self.current_marking = movement['target_marking'].copy()

            self.ax_petri.set_title("Token Movement Animation")
            return []

        # Create animation with more frames for smoother movement
        total_frames = len(token_movements) * 20  # 20 frames per movement
        if total_frames > 0:
            if is_cyclic:
                total_frames = total_frames * 4  # Show multiple cycles
            
            self.animation = FuncAnimation(
                self.fig_petri, update, init_func=init,
                frames=total_frames, interval=50, blit=False, repeat=is_cyclic)
            
            self.canvas_petri.draw()
        else:
            print("No frames to animate!")
    
    def replay_animation(self):
        if not self.marking_graph.edges:
            messagebox.showinfo("No Animation", "No transitions to replay. Add places, transitions, and run the exploration.")
            return
    
        # Stop the existing animation if it's still running
        if hasattr(self, 'animation') and self.animation is not None:
            try:
                if hasattr(self.animation, 'event_source') and self.animation.event_source is not None:
                    self.animation.event_source.stop()
            except:
                pass
            self.animation = None
    
        self.animate_marking(self.marking_graph)


    def visualize_marking(self, marking):
        # Similar to update_petri_visualization but uses the current marking
        G = nx.DiGraph()
        pos = {}
        
        for i, place in enumerate(self.places):
            G.add_node(place.name, type='place', tokens=marking[place.name])
            pos[place.name] = (i * 2, 0)
            
        for i, transition in enumerate(self.transitions):
            trans_name = f"T_{transition.name}"
            G.add_node(trans_name, type='transition')
            pos[trans_name] = (i * 2 + 1, 1)
            
            # Add edges
            for place in transition.inputs:
                G.add_edge(place.name, trans_name)
            for place in transition.outputs:
                G.add_edge(trans_name, place.name)

        nx.draw_networkx_nodes(G, pos, 
                             nodelist=[n for n, d in G.nodes(data=True) if d['type'] == 'place'],
                             node_color='lightblue',
                             node_size=1000)
        
        nx.draw_networkx_nodes(G, pos,
                             nodelist=[n for n, d in G.nodes(data=True) if d['type'] == 'transition'],
                             node_color='lightgreen',
                             node_shape='s',
                             node_size=800)

        # Draw edges with arrows
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos)
        
        # Draw tokens as small circles inside places
        for node, attr in G.nodes(data=True):
            if attr.get('type') == 'place':
                tokens = attr['tokens']
                if tokens > 0:
                    center_x, center_y = pos[node]
                    
                    # Calculate positions for tokens in a circular arrangement
                    for i in range(tokens):
                        angle = 2 * np.pi * i / max(tokens, 1)
                        if tokens == 1:
                            token_x, token_y = center_x, center_y
                        else:
                            token_x = center_x + 0.15 * 0.5 * np.cos(angle)
                            token_y = center_y + 0.15 * 0.5 * np.sin(angle)
                        self.ax_petri.plot(token_x, token_y, 'ko', markersize=8)

        self.ax_petri.set_title("Petri Net State")

    def explore_and_exit(self):
        self.start_exploration()
        # Removed root.quit() to keep the application running

    def start_exploration(self):
        if not self.is_master:
            return

        if not self.places or not self.transitions:
            messagebox.showinfo("Error", "Please add places and transitions first.")
            return

        initial_marking = {p.name: p.tokens for p in self.places}
        comm = MPI.COMM_WORLD
        size = comm.Get_size()

        visited = {str(initial_marking)}
        queue = [initial_marking]
        self.marking_graph = MarkingGraph()
        self.marking_graph.add_node(initial_marking)

        # For debugging
        print(f"Initial marking: {initial_marking}")
        print(f"Places: {[p.name for p in self.places]}")
        print(f"Transitions: {[t.name for t in self.transitions]}")

        while queue:
            current = queue.pop(0)
            self.add_to_history(f"Exploring marking: {current}")

            # Try each transition
            for transition in self.transitions:
                new_marking = transition.fire(current)
                if new_marking:
                    str_marking = str(new_marking)
                    if str_marking not in visited:
                        visited.add(str_marking)
                        queue.append(new_marking)
                        self.marking_graph.add_node(new_marking)
                        self.marking_graph.add_edge(current, transition.name, new_marking)
                        self.add_to_history(f"Found new marking via {transition.name}: {new_marking}")
                        print(f"Transition {transition.name} fired: {current} -> {new_marking}")

        if not self.marking_graph.edges:
            messagebox.showinfo("No Changes", "No valid transitions found. Check your Petri net configuration.")
            return

        print(f"Found {len(self.marking_graph.edges)} transitions")
        self.animate_marking(self.marking_graph)

    def hash_marking(self, marking, num_sites):
        return sum(marking.values()) % num_sites

    def worker_process(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        while True:
            data = comm.recv(source=0, tag=MPI.ANY_TAG)
            if data is None:
                break
            marking, transitions = data
            new_markings = self.explore_markings(marking, transitions)
            comm.send(new_markings, dest=0)

    def explore_markings(self, marking, transitions):
        result = []
        for t in transitions:
            new_marking = t.fire(marking)
            if new_marking:
                result.append(new_marking)
        return result

    def clear_all(self):
        """Clear all places, transitions, and reset the visualization"""
        # Safely stop any existing animation
        if hasattr(self, 'animation') and self.animation is not None:
            try:
                if hasattr(self.animation, 'event_source') and self.animation.event_source is not None:
                    self.animation.event_source.stop()
            except:
                pass
        self.animation = None
        
        # Clear all data
        self.places = []
        self.transitions = []
        self.marking_graph = MarkingGraph()
        self.transaction_history = []
        
        # Clear visualization
        if hasattr(self, 'ax_petri'):
            self.ax_petri.clear()
            self.canvas_petri.draw()
        
        # Clear Listboxes
        self.places_listbox.delete(0, tk.END)
        self.transitions_listbox.delete(0, tk.END)
        self.history_listbox.delete(0, tk.END)

        # Reset combobox values
        self.places_listbox.delete(0, tk.END)
        self.transitions_listbox.delete(0, tk.END)
        self.history_listbox.delete(0, tk.END)
        self.trans_input_listbox.delete(0, tk.END)  # Clear input places listbox
        self.trans_output_listbox.delete(0, tk.END)  # Clear output places listbox
        
        # Add to history
        self.add_to_history("Cleared all elements")


    def load_example(self):
        """Load an example Petri net"""
        # Clear existing elements
        self.clear_all()
        
        # Create example places
        example_places = [
            ("P1", 2),  # Initial place with 2 tokens
            ("P2", 0),  # Middle place
            ("P3", 0),  # Final place
            ("P4", 1),  # Another initial place with 1 token
        ]
        
        # Add places
        for name, tokens in example_places:
            place = Place(name, tokens)
            self.places.append(place)
            self.places_listbox.insert(tk.END, f"{name}: {tokens} tokens")
        
        # Create example transitions
        example_transitions = [
            ("T1", ["P1"], ["P2"]),      # P1 -> P2
            ("T2", ["P2", "P4"], ["P3"]), # P2 + P4 -> P3
            ("T3", ["P3"], ["P1", "P4"]), # P3 -> P1 + P4
        ]
        
        # Add transitions
        for name, inputs, outputs in example_transitions:
            input_places = [p for p in self.places if p.name in inputs]
            output_places = [p for p in self.places if p.name in outputs]
            transition = Transition(name, input_places, output_places)
            self.transitions.append(transition)
            self.transitions_listbox.insert(tk.END, f"{name}: {', '.join(inputs)} -> {', '.join(outputs)}")
        
        self.update_petri_visualization()
        self.add_to_history("Loaded example Petri net")
        messagebox.showinfo("Example Loaded", 
            "Loaded example Petri net:\n\n" +
            "Places:\n" +
            "- P1 (2 tokens)\n" +
            "- P2 (0 tokens)\n" +
            "- P3 (0 tokens)\n" +
            "- P4 (1 token)\n\n" +
            "Transitions:\n" +
            "- T1: P1 -> P2\n" +
            "- T2: P2 + P4 -> P3\n" +
            "- T3: P3 -> P1 + P4")

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        app = ModernPetriNetGUI(is_master=True)
        app.root.mainloop()
    else:
        worker = ModernPetriNetGUI(is_master=False)
        worker.worker_process()