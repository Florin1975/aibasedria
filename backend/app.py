from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from datetime import datetime, timedelta
import jwt
from werkzeug.security import generate_password_hash, check_password_hash

# Import modules
from document_processing import DocumentProcessor
from entity_extraction import EntityExtractor
from knowledge_graph import KnowledgeGraph
from analysis_engine import (
    CostBenefitAnalyzer,
    StakeholderImpactAnalyzer,
    RegulatoryBurdenCalculator,
    PolicyEffectivenessPredictor
)
from report_generator import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load configuration
app.config.from_pyfile('config.py')
SECRET_KEY = app.config.get('SECRET_KEY', 'default-secret-key')

# Initialize components
document_processor = DocumentProcessor()
entity_extractor = EntityExtractor()
knowledge_graph = KnowledgeGraph()
cost_benefit_analyzer = CostBenefitAnalyzer()
stakeholder_impact_analyzer = StakeholderImpactAnalyzer()
regulatory_burden_calculator = RegulatoryBurdenCalculator()
policy_effectiveness_predictor = PolicyEffectivenessPredictor()
report_generator = ReportGenerator()

# Authentication middleware
def token_required(f):
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
        
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        
        try:
            data = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            current_user = data['user_id']
        except:
            return jsonify({'message': 'Token is invalid'}), 401
        
        return f(current_user, *args, **kwargs)
    
    decorated.__name__ = f.__name__
    return decorated

# Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/api/auth/login', methods=['POST'])
def login():
    """User login endpoint"""
    data = request.get_json()
    
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'message': 'Missing username or password'}), 400
    
    # In a real implementation, validate against database
    # This is a simplified example
    if data.get('username') == 'admin' and data.get('password') == 'password':
        token = jwt.encode({
            'user_id': 'admin',
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, SECRET_KEY)
        
        return jsonify({'token': token})
    
    return jsonify({'message': 'Invalid credentials'}), 401

@app.route('/api/documents', methods=['POST'])
@token_required
def upload_document(current_user):
    """Upload a document for processing"""
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400
    
    # Process the document
    try:
        document_id = document_processor.process_document(file)
        return jsonify({'document_id': document_id, 'message': 'Document uploaded successfully'})
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return jsonify({'message': f'Error processing document: {str(e)}'}), 500

@app.route('/api/documents/<document_id>', methods=['GET'])
@token_required
def get_document(current_user, document_id):
    """Get document details"""
    try:
        document = document_processor.get_document(document_id)
        if document:
            return jsonify(document)
        return jsonify({'message': 'Document not found'}), 404
    except Exception as e:
        logger.error(f"Error retrieving document: {str(e)}")
        return jsonify({'message': f'Error retrieving document: {str(e)}'}), 500

@app.route('/api/analysis/cost-benefit', methods=['POST'])
@token_required
def analyze_cost_benefit(current_user):
    """Perform cost-benefit analysis"""
    data = request.get_json()
    
    if not data:
        return jsonify({'message': 'No data provided'}), 400
    
    try:
        result = cost_benefit_analyzer.analyze(data)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in cost-benefit analysis: {str(e)}")
        return jsonify({'message': f'Error in analysis: {str(e)}'}), 500

@app.route('/api/analysis/stakeholder-impact', methods=['POST'])
@token_required
def analyze_stakeholder_impact(current_user):
    """Analyze stakeholder impacts"""
    data = request.get_json()
    
    if not data:
        return jsonify({'message': 'No data provided'}), 400
    
    try:
        result = stakeholder_impact_analyzer.analyze(data)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in stakeholder impact analysis: {str(e)}")
        return jsonify({'message': f'Error in analysis: {str(e)}'}), 500

@app.route('/api/analysis/regulatory-burden', methods=['POST'])
@token_required
def calculate_regulatory_burden(current_user):
    """Calculate regulatory burden"""
    data = request.get_json()
    
    if not data:
        return jsonify({'message': 'No data provided'}), 400
    
    try:
        result = regulatory_burden_calculator.calculate(data)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in regulatory burden calculation: {str(e)}")
        return jsonify({'message': f'Error in calculation: {str(e)}'}), 500

@app.route('/api/analysis/policy-effectiveness', methods=['POST'])
@token_required
def predict_policy_effectiveness(current_user):
    """Predict policy effectiveness"""
    data = request.get_json()
    
    if not data:
        return jsonify({'message': 'No data provided'}), 400
    
    try:
        result = policy_effectiveness_predictor.predict(data)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error in policy effectiveness prediction: {str(e)}")
        return jsonify({'message': f'Error in prediction: {str(e)}'}), 500

@app.route('/api/reports/generate', methods=['POST'])
@token_required
def generate_report(current_user):
    """Generate RIA report"""
    data = request.get_json()
    
    if not data:
        return jsonify({'message': 'No data provided'}), 400
    
    try:
        report_id = report_generator.generate(data)
        return jsonify({'report_id': report_id, 'message': 'Report generated successfully'})
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return jsonify({'message': f'Error generating report: {str(e)}'}), 500

@app.route('/api/reports/<report_id>', methods=['GET'])
@token_required
def get_report(current_user, report_id):
    """Get report details"""
    try:
        report = report_generator.get_report(report_id)
        if report:
            return jsonify(report)
        return jsonify({'message': 'Report not found'}), 404
    except Exception as e:
        logger.error(f"Error retrieving report: {str(e)}")
        return jsonify({'message': f'Error retrieving report: {str(e)}'}), 500

@app.route('/api/reports/<report_id>/download', methods=['GET'])
@token_required
def download_report(current_user, report_id):
    """Download report in specified format"""
    format_type = request.args.get('format', 'pdf')
    
    try:
        report_file = report_generator.export_report(report_id, format_type)
        if report_file:
            return jsonify({'download_url': report_file})
        return jsonify({'message': 'Report not found'}), 404
    except Exception as e:
        logger.error(f"Error downloading report: {str(e)}")
        return jsonify({'message': f'Error downloading report: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8000)), debug=False)
