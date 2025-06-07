from datetime import datetime
import json
import os
import uuid
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class User(Base):
    """User model for authentication and authorization"""
    __tablename__ = 'users'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    first_name = Column(String(50))
    last_name = Column(String(50))
    role = Column(String(20), default='user')  # 'admin', 'user', 'viewer'
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    projects = relationship('Project', back_populates='user')
    
    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}')>"

class Project(Base):
    """Project model for organizing RIA assessments"""
    __tablename__ = 'projects'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(200), nullable=False)
    description = Column(Text)
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False)
    status = Column(String(20), default='draft')  # 'draft', 'in_progress', 'completed'
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship('User', back_populates='projects')
    regulatory_context = relationship('RegulatoryContext', uselist=False, back_populates='project')
    documents = relationship('Document', back_populates='project')
    stakeholders = relationship('Stakeholder', back_populates='project')
    impacts = relationship('Impact', back_populates='project')
    cost_benefits = relationship('CostBenefit', back_populates='project')
    reports = relationship('Report', back_populates='project')
    
    def __repr__(self):
        return f"<Project(title='{self.title}', status='{self.status}')>"

class RegulatoryContext(Base):
    """Regulatory context model for storing regulatory framework information"""
    __tablename__ = 'regulatory_contexts'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String(36), ForeignKey('projects.id'), nullable=False)
    regulation_title = Column(String(200))
    regulation_type = Column(String(50))  # 'new', 'amendment', 'repeal'
    jurisdiction = Column(String(100))
    sector = Column(String(100))
    problem_statement = Column(Text)
    objectives = Column(Text)
    alternatives_considered = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    project = relationship('Project', back_populates='regulatory_context')
    
    def __repr__(self):
        return f"<RegulatoryContext(regulation_title='{self.regulation_title}')>"

class Document(Base):
    """Document model for storing uploaded regulatory documents"""
    __tablename__ = 'documents'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String(36), ForeignKey('projects.id'), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(255), nullable=False)
    file_type = Column(String(50))  # 'pdf', 'docx', 'txt', etc.
    document_type = Column(String(50))  # 'regulation', 'impact_study', 'stakeholder_feedback', etc.
    processed = Column(Boolean, default=False)
    extracted_text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    project = relationship('Project', back_populates='documents')
    entities = relationship('Entity', back_populates='document')
    
    def __repr__(self):
        return f"<Document(filename='{self.filename}', document_type='{self.document_type}')>"

class Entity(Base):
    """Entity model for storing extracted entities from documents"""
    __tablename__ = 'entities'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String(36), ForeignKey('documents.id'), nullable=False)
    entity_type = Column(String(50))  # 'organization', 'regulation', 'requirement', 'deadline', etc.
    text = Column(String(255))
    start_pos = Column(Integer)
    end_pos = Column(Integer)
    metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship('Document', back_populates='entities')
    
    def __repr__(self):
        return f"<Entity(entity_type='{self.entity_type}', text='{self.text}')>"

class Stakeholder(Base):
    """Stakeholder model for storing stakeholder information"""
    __tablename__ = 'stakeholders'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String(36), ForeignKey('projects.id'), nullable=False)
    name = Column(String(100), nullable=False)
    category = Column(String(50))  # 'government', 'business', 'civil_society', 'individual', etc.
    description = Column(Text)
    contact_info = Column(Text)
    consultation_status = Column(String(50), default='not_started')  # 'not_started', 'in_progress', 'completed'
    consultation_method = Column(String(100))
    consultation_date = Column(DateTime)
    feedback = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    project = relationship('Project', back_populates='stakeholders')
    impacts = relationship('Impact', back_populates='stakeholder')
    
    def __repr__(self):
        return f"<Stakeholder(name='{self.name}', category='{self.category}')>"

class Impact(Base):
    """Impact model for storing impact assessment information"""
    __tablename__ = 'impacts'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String(36), ForeignKey('projects.id'), nullable=False)
    stakeholder_id = Column(String(36), ForeignKey('stakeholders.id'))
    impact_type = Column(String(50))  # 'economic', 'social', 'environmental', 'administrative', 'competitive'
    description = Column(Text)
    magnitude = Column(Float)  # -1 to 1, negative to positive
    likelihood = Column(Float)  # 0 to 1
    timeframe = Column(String(50))  # 'short_term', 'medium_term', 'long_term'
    evidence = Column(Text)
    mitigation_measures = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    project = relationship('Project', back_populates='impacts')
    stakeholder = relationship('Stakeholder', back_populates='impacts')
    
    def __repr__(self):
        return f"<Impact(impact_type='{self.impact_type}', magnitude={self.magnitude})>"

class CostBenefit(Base):
    """Cost-benefit model for storing cost-benefit analysis information"""
    __tablename__ = 'cost_benefits'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String(36), ForeignKey('projects.id'), nullable=False)
    item_type = Column(String(10), nullable=False)  # 'cost' or 'benefit'
    name = Column(String(100), nullable=False)
    description = Column(Text)
    category = Column(String(50))  # 'direct', 'indirect', 'compliance', 'administrative', etc.
    value = Column(Float)
    is_recurring = Column(Boolean, default=False)
    recurrence_period = Column(String(50))  # 'annual', 'quarterly', 'monthly', etc.
    affected_stakeholders = Column(Text)
    monetized = Column(Boolean, default=True)
    calculation_method = Column(Text)
    assumptions = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    project = relationship('Project', back_populates='cost_benefits')
    
    def __repr__(self):
        return f"<CostBenefit(item_type='{self.item_type}', name='{self.name}', value={self.value})>"

class Report(Base):
    """Report model for storing generated RIA reports"""
    __tablename__ = 'reports'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    project_id = Column(String(36), ForeignKey('projects.id'), nullable=False)
    title = Column(String(200), nullable=False)
    version = Column(String(20), default='1.0')
    format = Column(String(20), default='pdf')  # 'pdf', 'docx', 'html', 'json'
    file_path = Column(String(255))
    generated_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    project = relationship('Project', back_populates='reports')
    
    def __repr__(self):
        return f"<Report(title='{self.title}', version='{self.version}', format='{self.format
