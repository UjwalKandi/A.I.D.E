import os
import sys
import sqlite3
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import Image, ImageTk
import threading
from scipy import stats
import tempfile

# Set default USER_AGENT
os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

# Updated imports for the latest versions
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, WebBaseLoader

class ChatAgent:
    def __init__(self, model_name="llama3.2:1b"):
        self.model_name = model_name
        self._initialize_llm()
        
    def _initialize_llm(self):
        """Separate initialization to allow reconnection attempts"""
        try:
            self.llm = ChatOllama(model=self.model_name)
            self.vector_store = VectorStore()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChatAgent: {str(e)}")
    
    def get_response(self, query, context=""):
        if not query.strip():
            return "Error: Empty query provided"
            
        max_retries = 3
        for attempt in range(max_retries):
            try:
                prompt = f"Context: {context}\n\nQuery: {query}\n\nResponse:"
                response = self.llm.invoke(prompt)
                
                if not response or not hasattr(response, 'content'):
                    return "Error: Invalid response from language model"
                    
                return response.content
                
            except ConnectionError:
                if attempt < max_retries - 1:
                    self._initialize_llm()  # Try to reconnect
                    continue
                return "Error: Failed to connect to LLM service after multiple attempts"
            except Exception as e:
                return f"Error: Unexpected error while processing query - {str(e)}"

class VectorStore:
    def __init__(self):
        try:
            self.embeddings = OllamaEmbeddings(model="llama2")
            self.db = Chroma(persist_directory="./chroma_db", 
                           embedding_function=self.embeddings)
        except Exception as e:
            print(f"Error initializing VectorStore: {e}")
            raise RuntimeError(f"Failed to initialize VectorStore: {str(e)}")
        
    def validate_url(self, url):
        """Validate URL format and accessibility."""
        if not url.strip():
            return False, "Empty URL provided"
            
        if not url.startswith(('http://', 'https://')):
            return False, "Invalid URL format - must start with http:// or https://"
            
        return True, "URL is valid"
        
    def add_url(self, url):
        try:
            # Validate URL first
            is_valid, message = self.validate_url(url)
            if not is_valid:
                return message
                
            # Attempt to load and process the URL
            loader = WebBaseLoader(url)
            documents = loader.load()
            
            if not documents:
                return "Error: No content found at URL"
                
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            
            if not splits:
                return "Error: Failed to split document content"
                
            self.db.add_documents(splits)
            return True
            
        except ConnectionError as e:
            return f"Error: Failed to connect to URL - {str(e)}"
        except TimeoutError as e:
            return f"Error: Connection timed out - {str(e)}"
        except Exception as e:
            return f"Error: Failed to process URL - {str(e)}"
            
    def query(self, query_text, k=3):
        if not query_text.strip():
            return []
            
        try:
            results = self.db.similarity_search(query_text, k=k)
            
            if not results:
                print("Warning: No similar documents found")
                
            return results
            
        except ValueError as e:
            print(f"Error: Invalid query parameters - {e}")
            return []
        except Exception as e:
            print(f"Error querying vector store: {e}")
            return []

# Updated TimeSeriesAnalyzer class
class TimeSeriesAnalyzer:
    def __init__(self):
        self.data = None
        self.analysis_results = None
        
    def load_data(self, df):
        try:
            # Convert to datetime if possible
            time_cols = [col for col in df.columns 
                        if 'time' in col.lower() or 'date' in col.lower()]
            if time_cols:
                df[time_cols[0]] = pd.to_datetime(df[time_cols[0]])
                df.set_index(time_cols[0], inplace=True)
            
            self.data = df
            return True
        except Exception as e:
            return f"Error loading data: {str(e)}"
            
    def detect_anomalies(self):
        if self.data is None:
            return "No data loaded"
            
        try:
            results = {}
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                series = self.data[col]
                
                # Calculate rolling statistics
                rolling_mean = series.rolling(window=5).mean()
                rolling_std = series.rolling(window=5).std()
                
                # Z-score based anomaly detection
                z_scores = np.abs(stats.zscore(series, nan_policy='omit'))
                anomalies = np.where(z_scores > 3)[0]
                
                # Detect sudden changes using rolling difference
                diff = series.diff()
                mean_diff = diff.mean()
                std_diff = diff.std()
                sudden_changes = np.where(np.abs(diff - mean_diff) > 2 * std_diff)[0]
                
                results[col] = {
                    'anomalies': anomalies,
                    'sudden_changes': sudden_changes,
                    'rolling_mean': rolling_mean,
                    'rolling_std': rolling_std
                }
            
            self.analysis_results = results
            return True
            
        except Exception as e:
            return f"Error in anomaly detection: {str(e)}"
    
    def create_analysis_plots(self):
        """Creates two separate plots: one for the GUI and one for saving"""
        if self.data is None or self.analysis_results is None:
            return None, None
            
        try:
            plots = []
            for plot_type in ['gui', 'file']:
                # Create figure
                num_cols = len(self.analysis_results)
                fig, axes = plt.subplots(num_cols, 1, figsize=(12, 4*num_cols))
                if num_cols == 1:
                    axes = [axes]
                
                for idx, (col_name, results) in enumerate(self.analysis_results.items()):
                    ax = axes[idx]
                    series = self.data[col_name]
                    
                    # Plot original data
                    ax.plot(self.data.index, series, 'b-', label='Data', alpha=0.5)
                    
                    # Plot rolling mean
                    ax.plot(self.data.index, results['rolling_mean'], 
                           'g-', label='Rolling Mean', alpha=0.7)
                    
                    # Plot anomalies
                    if len(results['anomalies']) > 0:
                        ax.scatter(self.data.index[results['anomalies']],
                                  series.iloc[results['anomalies']],
                                  color='red', label='Anomalies', s=50)
                    
                    # Plot sudden changes
                    if len(results['sudden_changes']) > 0:
                        ax.scatter(self.data.index[results['sudden_changes']],
                                  series.iloc[results['sudden_changes']],
                                  color='orange', label='Sudden Changes', 
                                  marker='^', s=50)
                    
                    ax.set_title(f'Analysis of {col_name}')
                    ax.legend()
                    ax.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                plots.append(fig)
            
            # Save the second plot to file
            plots[1].savefig('analysis_results.png', bbox_inches='tight', dpi=300)
            plt.close(plots[1])
            
            return plots[0], 'analysis_results.png'
            
        except Exception as e:
            print(f"Error in create_analysis_plots: {e}")
            return None, None

class LogAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("A.I.D.E")
        self.root.geometry("1200x800")
        
        self.ts_analyzer = None
        self.chat_agent = None
        self.canvas = None
        self.figure = None
        self.plot_image = None
        
        try:
            self.ts_analyzer = TimeSeriesAnalyzer()
            self.chat_agent = ChatAgent()
            self.setup_gui()
        except Exception as e:
            messagebox.showerror("Initialization Error", str(e))
            raise

    def cleanup(self):
        """Clean up resources before closing."""
        try:
            plt.close('all')
            if hasattr(self, 'plot_image'):
                del self.plot_image
            if os.path.exists('analysis_results.png'):
                os.remove('analysis_results.png')
        except Exception as e:
            print(f"Cleanup error: {e}")

    def on_closing(self):
        """Handle window closing event."""
        self.cleanup()
        self.root.destroy()

    def setup_gui(self):
        """Initialize all GUI components."""
        # Create main container for split view
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True)
        
        # Left panel
        left_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(left_frame, weight=1)
        
        # Chat interface
        chat_frame = ttk.LabelFrame(left_frame, text="Chat Interface")
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Model selection
        model_frame = ttk.Frame(chat_frame)
        model_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(model_frame, text="Model:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value="llama3.2:1b")
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var,
                                 values=["llama3.2:1b", "llama3.2:3b"],
                                 state="readonly")
        model_combo.pack(side=tk.LEFT, padx=5)
        
        # Chat area
        self.chat_area = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD,
                                                 height=20)
        self.chat_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Input area
        input_frame = ttk.Frame(chat_frame)
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        self.chat_input = ttk.Entry(input_frame)
        self.chat_input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.chat_input.bind('<Return>', lambda e: self.send_message())
        ttk.Button(input_frame, text="Send",
                  command=self.send_message).pack(side=tk.RIGHT, padx=5)
        
        # Right panel
        right_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(right_frame, weight=1)
        
        # File analysis section
        file_frame = ttk.LabelFrame(right_frame, text="Data Analysis")
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # File selection
        file_select_frame = ttk.Frame(file_frame)
        file_select_frame.pack(fill=tk.X, padx=5, pady=5)
        self.file_path = ttk.Entry(file_select_frame)
        self.file_path.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(file_select_frame, text="Browse",
                  command=self.browse_file).pack(side=tk.RIGHT, padx=5)
        
        # Knowledge base section
        kb_frame = ttk.LabelFrame(right_frame, text="Knowledge Hub")
        kb_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # URL input
        url_frame = ttk.Frame(kb_frame)
        url_frame.pack(fill=tk.X, padx=5, pady=5)
        self.url_entry = ttk.Entry(url_frame)
        self.url_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.url_entry.bind('<Return>', lambda e: self.add_url())
        ttk.Button(url_frame, text="Add URL",
                  command=self.add_url).pack(side=tk.RIGHT, padx=5)
        
        # Document list
        self.doc_list = ttk.Treeview(kb_frame, columns=("url",),
                                   show="headings", height=5)
        self.doc_list.heading("url", text="Added Documents")
        self.doc_list.pack(fill=tk.X, padx=5, pady=5)
        
        # Analysis visualization
        viz_frame = ttk.LabelFrame(right_frame, text="Analysis Results")
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Status indicator
        status_frame = ttk.Frame(viz_frame)
        status_frame.pack(fill=tk.X, padx=5, pady=5)
        self.status_indicator = ttk.Label(status_frame, text="●",
                                       foreground="gray")
        self.status_indicator.pack(side=tk.LEFT)
        self.status_label = ttk.Label(status_frame,
                                    text="Ready for analysis")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # Canvas for plot
        self.canvas_frame = ttk.Frame(viz_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_status(self, message, color="gray"):
        """Update the status indicator and message."""
        self.status_indicator.config(foreground=color)
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def send_message(self):
        """Handle sending chat messages."""
        message = self.chat_input.get().strip()
        if message:
            self.chat_area.insert(tk.END, f"You: {message}\n")
            self.chat_input.delete(0, tk.END)
            
            try:
                current_model = self.model_var.get()
                if self.chat_agent.llm.model != current_model:
                    self.chat_agent = ChatAgent(current_model)
                
                context_docs = self.chat_agent.vector_store.query(message)
                context = "\n".join([doc.page_content for doc in context_docs])
                
                response = self.chat_agent.get_response(message, context)
                self.chat_area.insert(tk.END, f"Assistant: {response}\n\n")
                self.chat_area.see(tk.END)
            except Exception as e:
                self.chat_area.insert(tk.END,
                                    f"Error: {str(e)}\n\n")
                self.chat_area.see(tk.END)

    def add_url(self):
        """Handle adding URLs to the knowledge base."""
        url = self.url_entry.get().strip()
        if url:
            self.update_status("Adding URL...", "yellow")
            result = self.chat_agent.vector_store.add_url(url)
            if result is True:
                self.doc_list.insert("", "end", values=(url,))
                self.url_entry.delete(0, tk.END)
                self.update_status("URL added successfully", "green")
            else:
                self.update_status("Failed to add URL", "red")
                messagebox.showerror("Error", f"Failed to add URL: {result}")

    def browse_file(self):
        """Handle file selection dialog."""
        filename = filedialog.askopenfilename(
            filetypes=[
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx;*.xls"),
                ("Text files", "*.txt"),
                ("Log files", "*.log"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.file_path.delete(0, tk.END)
            self.file_path.insert(0, filename)
            self.analyze_file(filename)

    def analyze_file(self, filename):
        """Handle file analysis and visualization."""
        try:
            self.update_status("Loading data...", "yellow")
            
            # Load data based on file type
            file_ext = filename.lower().split('.')[-1]
            if file_ext in ['xlsx', 'xls']:
                df = pd.read_excel(filename)
            else:
                df = pd.read_csv(filename)
            
            # Load and analyze data
            if self.ts_analyzer.load_data(df):
                self.update_status("Analyzing data...", "yellow")
                if self.ts_analyzer.detect_anomalies():
                    # Generate both plots
                    gui_figure, saved_plot = self.ts_analyzer.create_analysis_plots()
                    
                    if gui_figure and saved_plot:
                        # Clear existing canvas
                        for widget in self.canvas_frame.winfo_children():
                            widget.destroy()
                        
                        # Create new canvas with the GUI figure
                        self.canvas = FigureCanvasTkAgg(gui_figure, self.canvas_frame)
                        self.canvas.draw()
                        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                        
                        # Add toolbar for plot interaction
                        toolbar_frame = ttk.Frame(self.canvas_frame)
                        toolbar_frame.pack(fill=tk.X)
                        toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
                        toolbar.update()
                        
                        self.update_status("Analysis complete", "green")
                        messagebox.showinfo("Success", 
                                          f"Analysis complete!\nResults saved to {saved_plot}")
                    else:
                        self.update_status("Failed to create visualizations", "red")
                else:
                    self.update_status("Analysis failed", "red")
            else:
                self.update_status("Failed to load data", "red")
                
        except Exception as e:
            self.update_status(f"Error: {str(e)}", "red")
            messagebox.showerror("Analysis Error", str(e))

def main():
    """Main entry point of the application."""
    try:
        root = tk.Tk()
        app = LogAnalyzerApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
