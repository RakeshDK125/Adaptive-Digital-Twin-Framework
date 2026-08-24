import pytest
from app.services.decision.engine import DecisionIntelligenceEngine
from app.domain.twin.models import MachineModel, VirtualRepresentation

@pytest.fixture
def decision_engine():
    return DecisionIntelligenceEngine()

def test_decision_fusion_logic(decision_engine):
    machine = MachineModel("Test Machine")
    # Simulate high wear to force a high risk score
    machine.parameters["wear"] = 0.8 
    virtual_rep = VirtualRepresentation(machine)
    
    report = decision_engine.generate_report(machine.id, virtual_rep)
    
    assert "risk_score" in report
    assert "explainability" in report
    assert "alternative_actions" in report
    assert "confidence_score" in report
    
    # With wear at 0.8, risk score should be elevated
    assert report["risk_score"] > 50.0
    
    # We should have a failure prediction message
    assert "High Risk" in report["failure_prediction"] or "Medium Risk" in report["failure_prediction"]
    
    # Check explainability text formatting
    assert "Decision driven by" in report["explainability"]
