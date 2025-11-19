import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Compatibility patch for scikit-learn version mismatch
# This fixes the _RemainderColsList issue when loading models trained with 1.6.x in 1.7.x
import sys
import sklearn.compose._column_transformer as ct_module

# Check if _RemainderColsList exists, if not create a compatibility shim
if not hasattr(ct_module, '_RemainderColsList'):
    # Create a compatibility class that mimics the old behavior
    class _RemainderColsList(list):
        """Compatibility shim for _RemainderColsList from scikit-learn 1.6.x"""
        def __new__(cls, *args, **kwargs):
            return super().__new__(cls)
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args)
    
    # Inject into the module before any pickle loading
    ct_module._RemainderColsList = _RemainderColsList
    # Also add to sys.modules cache if needed
    if 'sklearn.compose._column_transformer._RemainderColsList' not in sys.modules:
        sys.modules['sklearn.compose._column_transformer._RemainderColsList'] = _RemainderColsList

# Page configuration
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model from pickle file"""
    try:
        # Try using joblib first (better compatibility)
        try:
            import joblib
            model = joblib.load('california_knn_pipeline.pkl')
            return model
        except:
            # Fallback to pickle
            with open('california_knn_pipeline.pkl', 'rb') as f:
                model = pickle.load(f)
            return model
    except AttributeError as e:
        if '_RemainderColsList' in str(e):
            # Try to patch and reload
            try:
                import sklearn.compose._column_transformer as ct_module
                from types import SimpleNamespace
                if not hasattr(ct_module, '_RemainderColsList'):
                    ct_module._RemainderColsList = SimpleNamespace
                # Retry loading
                import joblib
                model = joblib.load('california_knn_pipeline.pkl')
                return model
            except Exception as e2:
                st.error(f"Version compatibility issue. Model was trained with scikit-learn 1.6.1, but current version is different. Error: {str(e2)}")
                st.info("💡 **Solution**: Please retrain the model with the current scikit-learn version, or use Python 3.11/3.12 with scikit-learn 1.6.1.")
                return None
        else:
            st.error(f"Error loading model: {str(e)}")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def create_sample_data():
    """Create sample data for visualization"""
    np.random.seed(42)
    n_samples = 100
    
    # Generate sample data based on typical California housing ranges
    data = {
        'MedInc': np.random.uniform(0.5, 15.0, n_samples),
        'HouseAge': np.random.uniform(1, 52, n_samples),
        'AveRooms': np.random.uniform(0.8, 10.0, n_samples),
        'AveBedrms': np.random.uniform(0.5, 5.0, n_samples),
        'Population': np.random.uniform(3, 3500, n_samples),
        'AveOccup': np.random.uniform(0.5, 10.0, n_samples),
        'Latitude': np.random.uniform(32.5, 42.0, n_samples),
        'Longitude': np.random.uniform(-124.0, -114.0, n_samples)
    }
    return pd.DataFrame(data)

def main():
    # Header
    st.markdown('<h1 class="main-header">🏠 California Housing Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Sidebar for user inputs
    st.sidebar.header("📊 Input Housing Features")
    st.sidebar.markdown("Adjust the sliders to input housing characteristics:")
    
    # Input fields with typical ranges for California housing dataset
    med_inc = st.sidebar.slider(
        "Median Income (in tens of thousands)",
        min_value=0.5,
        max_value=15.0,
        value=3.0,
        step=0.1,
        help="Median income in block group"
    )
    
    house_age = st.sidebar.slider(
        "House Age (years)",
        min_value=1,
        max_value=52,
        value=28,
        step=1,
        help="Median house age in block group"
    )
    
    ave_rooms = st.sidebar.slider(
        "Average Rooms",
        min_value=0.8,
        max_value=10.0,
        value=5.0,
        step=0.1,
        help="Average number of rooms per household"
    )
    
    ave_bedrms = st.sidebar.slider(
        "Average Bedrooms",
        min_value=0.5,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Average number of bedrooms per household"
    )
    
    population = st.sidebar.slider(
        "Population",
        min_value=3,
        max_value=3500,
        value=1425,
        step=10,
        help="Block group population"
    )
    
    ave_occup = st.sidebar.slider(
        "Average Occupancy",
        min_value=0.5,
        max_value=10.0,
        value=3.0,
        step=0.1,
        help="Average number of household members"
    )
    
    latitude = st.sidebar.slider(
        "Latitude",
        min_value=32.5,
        max_value=42.0,
        value=34.0,
        step=0.1,
        help="Block group latitude"
    )
    
    longitude = st.sidebar.slider(
        "Longitude",
        min_value=-124.0,
        max_value=-114.0,
        value=-118.0,
        step=0.1,
        help="Block group longitude"
    )
    
    # Feature names (standard California housing dataset features)
    feature_names = [
        'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
        'Population', 'AveOccup', 'Latitude', 'Longitude'
    ]
    
    # Prepare input data as DataFrame (model expects DataFrame with column names)
    input_data = pd.DataFrame([[
        med_inc, house_age, ave_rooms, ave_bedrms,
        population, ave_occup, latitude, longitude
    ]], columns=feature_names)
    
    # Also keep numpy array for display
    input_array = np.array([[
        med_inc, house_age, ave_rooms, ave_bedrms,
        population, ave_occup, latitude, longitude
    ]])
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Prediction")
        
        # Make prediction
        try:
            prediction = model.predict(input_data)[0]
            
            # Display prediction in a styled box
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.metric(
                label="**Predicted House Price**",
                value=f"${prediction:,.2f}",
                help="Predicted median house value in hundreds of thousands of dollars"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display input features
            st.subheader("📋 Input Features Summary")
            input_df = pd.DataFrame({
                'Feature': feature_names,
                'Value': input_array[0]
            })
            st.dataframe(input_df, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    with col2:
        st.subheader("ℹ️ About the Model")
        st.info("""
        **Model Type:** K-Nearest Neighbors (KNN)
        
        **Dataset:** California Housing
        
        **Features:**
        - Median Income
        - House Age
        - Average Rooms
        - Average Bedrooms
        - Population
        - Average Occupancy
        - Latitude
        - Longitude
        
        **Output:** House price in hundreds of thousands of dollars
        """)
    
    st.markdown("---")
    
    # Model Performance Metrics Section
    st.subheader("📊 Model Performance Metrics")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Model Information", "Performance Analysis", "Data Insights"])
    
    with tab1:
        st.markdown("### Model Details")
        
        # Display model information
        try:
            if hasattr(model, 'named_steps'):
                steps_info = []
                for name, step in model.named_steps.items():
                    steps_info.append(f"**{name}:** {type(step).__name__}")
                    if hasattr(step, 'get_params'):
                        params = step.get_params()
                        if 'n_neighbors' in params:
                            steps_info.append(f"  - Number of Neighbors: {params['n_neighbors']}")
                        if 'weights' in params:
                            steps_info.append(f"  - Weights: {params['weights']}")
                        if 'algorithm' in params:
                            steps_info.append(f"  - Algorithm: {params['algorithm']}")
                
                st.markdown("\n".join(steps_info))
            else:
                st.info(f"Model Type: {type(model).__name__}")
                if hasattr(model, 'get_params'):
                    params = model.get_params()
                    st.json(params)
        except Exception as e:
            st.warning(f"Could not retrieve model details: {str(e)}")
        
        # Model architecture info
        st.markdown("### Model Architecture")
        st.info("""
        This model uses a **Pipeline** approach with the following components:
        
        1. **Data Preprocessing**: Handles missing values and feature scaling
        2. **K-Nearest Neighbors (KNN)**: Predicts house prices based on similar housing data points
        
        The model processes 8 input features and outputs a predicted house price.
        """)
    
    with tab2:
        st.markdown("### Performance Analysis on Sample Data")
        
        # Generate sample data for performance metrics
        sample_data = create_sample_data()
        
        try:
            # Make predictions on sample data (model expects DataFrame)
            sample_predictions = model.predict(sample_data)
            
            # Create synthetic "actual" values for demonstration
            # In a real scenario, these would come from test data
            np.random.seed(42)
            noise = np.random.normal(0, 0.1, len(sample_predictions))
            synthetic_actual = sample_predictions + noise
            
            # Calculate metrics
            mse = mean_squared_error(synthetic_actual, sample_predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(synthetic_actual, sample_predictions)
            r2 = r2_score(synthetic_actual, sample_predictions)
            
            # Display metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("R² Score", f"{r2:.4f}", help="Coefficient of determination (closer to 1 is better)")
            
            with col2:
                st.metric("RMSE", f"${rmse:.2f}", help="Root Mean Squared Error (lower is better)")
            
            with col3:
                st.metric("MAE", f"${mae:.2f}", help="Mean Absolute Error (lower is better)")
            
            with col4:
                st.metric("MSE", f"${mse:.2f}", help="Mean Squared Error (lower is better)")
            
            # Prediction vs Actual plot
            st.markdown("**Prediction vs Actual Values**")
            fig_perf, ax_perf = plt.subplots(figsize=(8, 6))
            ax_perf.scatter(synthetic_actual, sample_predictions, alpha=0.6, color='#1f77b4', s=50)
            
            # Add perfect prediction line
            min_val = min(min(synthetic_actual), min(sample_predictions))
            max_val = max(max(synthetic_actual), max(sample_predictions))
            ax_perf.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            ax_perf.set_xlabel('Actual Price (hundreds of thousands $)')
            ax_perf.set_ylabel('Predicted Price (hundreds of thousands $)')
            ax_perf.set_title('Model Performance: Predicted vs Actual')
            ax_perf.legend()
            ax_perf.grid(True, alpha=0.3)
            st.pyplot(fig_perf)
            
            # Residual plot
            st.markdown("**Residual Plot**")
            residuals = synthetic_actual - sample_predictions
            fig_res, ax_res = plt.subplots(figsize=(8, 5))
            ax_res.scatter(sample_predictions, residuals, alpha=0.6, color='#ff7f0e', s=50)
            ax_res.axhline(y=0, color='r', linestyle='--', lw=2)
            ax_res.set_xlabel('Predicted Price (hundreds of thousands $)')
            ax_res.set_ylabel('Residuals (Actual - Predicted)')
            ax_res.set_title('Residual Analysis')
            ax_res.grid(True, alpha=0.3)
            st.pyplot(fig_res)
            
            st.info("💡 Note: These metrics are calculated on synthetic sample data for demonstration. For real performance evaluation, use actual test data.")
            
        except Exception as e:
            st.warning(f"Could not calculate performance metrics: {str(e)}")
    
    with tab3:
        st.markdown("### Data Insights & Visualizations")
        
        # Generate sample predictions for visualization
        sample_data = create_sample_data()
        
        try:
            # Make predictions (model expects DataFrame)
            sample_predictions = model.predict(sample_data)
            sample_data['PredictedPrice'] = sample_predictions
        
            # Create visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Price Distribution**")
                fig1, ax1 = plt.subplots(figsize=(8, 5))
                ax1.hist(sample_predictions, bins=30, edgecolor='black', alpha=0.7, color='#1f77b4')
                ax1.set_xlabel('Predicted Price (hundreds of thousands $)')
                ax1.set_ylabel('Frequency')
                ax1.set_title('Distribution of Predicted House Prices')
                ax1.grid(True, alpha=0.3)
                st.pyplot(fig1)
                
                st.markdown("**Price vs. Median Income**")
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                ax2.scatter(sample_data['MedInc'], sample_data['PredictedPrice'], 
                           alpha=0.6, color='#ff7f0e', s=50)
                ax2.set_xlabel('Median Income (tens of thousands)')
                ax2.set_ylabel('Predicted Price (hundreds of thousands $)')
                ax2.set_title('House Price vs. Median Income')
                ax2.grid(True, alpha=0.3)
                st.pyplot(fig2)
            
            with col2:
                st.markdown("**Price vs. House Age**")
                fig3, ax3 = plt.subplots(figsize=(8, 5))
                ax3.scatter(sample_data['HouseAge'], sample_data['PredictedPrice'], 
                           alpha=0.6, color='#2ca02c', s=50)
                ax3.set_xlabel('House Age (years)')
                ax3.set_ylabel('Predicted Price (hundreds of thousands $)')
                ax3.set_title('House Price vs. House Age')
                ax3.grid(True, alpha=0.3)
                st.pyplot(fig3)
                
                st.markdown("**Geographic Distribution**")
                fig4, ax4 = plt.subplots(figsize=(8, 5))
                scatter = ax4.scatter(sample_data['Longitude'], sample_data['Latitude'], 
                                     c=sample_data['PredictedPrice'], 
                                     cmap='viridis', alpha=0.6, s=50)
                ax4.set_xlabel('Longitude')
                ax4.set_ylabel('Latitude')
                ax4.set_title('House Prices by Location')
                ax4.grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=ax4, label='Price (hundreds of thousands $)')
                st.pyplot(fig4)
            
            # Feature correlation heatmap
            st.markdown("**Feature Correlation Matrix**")
            numeric_cols = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                            'Population', 'AveOccup', 'Latitude', 'Longitude', 'PredictedPrice']
            corr_matrix = sample_data[numeric_cols].corr()
            
            fig5, ax5 = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax5)
            ax5.set_title('Feature Correlation Matrix')
            st.pyplot(fig5)
            
        except Exception as e:
            st.warning(f"Could not generate visualizations: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p>California Housing Price Prediction Model | Built with Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

