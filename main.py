from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import os
from sqlalchemy.engine import URL

from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, Boolean,
    Float, Text, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship

# ----------------- Config Banco -----------------

DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")

# Monta a URL de forma segura, sem quebrar quando a senha tem @, !, etc.
db_url = URL.create(
    drivername="mysql+pymysql",
    username=DB_USER,
    password=DB_PASS,
    host=DB_HOST,
    port=3306,
    database=DB_NAME,
)

engine = create_engine(db_url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ----------------- Models (espelho das tabelas) -----------------

class Employee(Base):
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, index=True)
    employee_code = Column(String(50), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=False)
    email = Column(String(255))
    department = Column(String(255))
    face_photo_url = Column(Text)
    face_photo_key = Column(Text)
    face_descriptor = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)


class FridgeSession(Base):
    __tablename__ = "fridge_sessions"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    device_id = Column(String(255), nullable=False)
    opened_at = Column(DateTime)
    closed_at = Column(DateTime, nullable=True)
    capture_before_id = Column(Integer, nullable=True)
    capture_after_id = Column(Integer, nullable=True)
    status = Column(String(50), default="open")
    notes = Column(Text)

    employee = relationship("Employee")


class RecognitionLog(Base):
    __tablename__ = "recognition_logs"

    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(Integer, nullable=True)
    device_id = Column(String(255))
    confidence = Column(Float)
    success = Column(Boolean)
    error_message = Column(Text, nullable=True)
    timestamp = Column(DateTime)


class VisionItem(Base):
    __tablename__ = "vision_items"

    id = Column(Integer, primary_key=True, index=True)
    capture_id = Column(Integer, index=True)
    label = Column(String(255))
    quantity = Column(Integer)


class ConsumptionEvent(Base):
    __tablename__ = "consumption_events"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, index=True)
    employee_id = Column(Integer, index=True)
    product_label = Column(String(255))
    quantity = Column(Integer)
    created_at = Column(DateTime)


# ----------------- Schemas (Pydantic) -----------------

class EmployeeCreate(BaseModel):
    employee_code: str
    name: str
    email: Optional[str] = None
    department: Optional[str] = None
    face_photo_base64: str
    face_descriptor: str


class EmployeeOut(BaseModel):
    id: int
    employee_code: str
    name: str
    email: Optional[str]
    department: Optional[str]
    face_photo_url: Optional[str]
    face_descriptor: Optional[str]

    class Config:
        orm_mode = True


class RecognitionLogCreate(BaseModel):
    employee_code: Optional[str] = None
    device_id: str
    confidence: Optional[float] = None
    success: bool
    error_message: Optional[str] = None


class SessionStartRequest(BaseModel):
    employee_code: str
    device_id: str


class SessionStartResponse(BaseModel):
    success: bool
    session_id: int
    employee: EmployeeOut


class SessionCloseRequest(BaseModel):
    session_id: int
    capture_before_id: Optional[int] = None
    capture_after_id: Optional[int] = None


class SessionListItem(BaseModel):
    id: int
    employee_id: int
    employee_code: str
    employee_name: str
    device_id: str
    opened_at: Optional[datetime]
    closed_at: Optional[datetime]
    capture_before_id: Optional[int]
    capture_after_id: Optional[int]
    status: str
    notes: Optional[str]

    class Config:
        orm_mode = True


# ----------------- FastAPI app -----------------

app = FastAPI(title="IceVision Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # depois a gente fecha
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Helpers -----------------

def now():
    return datetime.utcnow()


# ----------------- Endpoints -----------------

@app.post("/api/employee_create")
def create_employee(payload: EmployeeCreate, db: Session = Depends(get_db)):
    # checa código duplicado
    existing = db.query(Employee).filter(
        Employee.employee_code == payload.employee_code
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail="Código de colaborador já existe")

    # TODO: salvar foto em disco ou S3 (por enquanto vamos só ignorar a base64)
    photo_url = None
    photo_key = None

    emp = Employee(
        employee_code=payload.employee_code,
        name=payload.name,
        email=payload.email,
        department=payload.department,
        face_photo_url=photo_url,
        face_photo_key=photo_key,
        face_descriptor=payload.face_descriptor,
        is_active=True,
        created_at=now(),
        updated_at=now(),
    )
    db.add(emp)
    db.commit()
    db.refresh(emp)

    return {
        "success": True,
        "employee_id": emp.id,
        "photo_url": emp.face_photo_url,
    }


@app.get("/api/employees_list", response_model=dict)
def list_employees(db: Session = Depends(get_db)):
    emps = db.query(Employee).order_by(Employee.name).all()
    return {
        "success": True,
        "employees": [
            EmployeeOut.from_orm(e) for e in emps
        ],
    }


@app.post("/api/recognition_log")
def recognition_log(payload: RecognitionLogCreate, db: Session = Depends(get_db)):
    employee_id = None
    if payload.employee_code:
        emp = (
            db.query(Employee)
            .filter(Employee.employee_code == payload.employee_code)
            .first()
        )
        if emp:
            employee_id = emp.id

    log = RecognitionLog(
        employee_id=employee_id,
        device_id=payload.device_id,
        confidence=payload.confidence or 0.0,
        success=payload.success,
        error_message=payload.error_message,
        timestamp=now(),
    )
    db.add(log)
    db.commit()
    return {"success": True}


@app.post("/api/session_start", response_model=SessionStartResponse)
def session_start(payload: SessionStartRequest, db: Session = Depends(get_db)):
    emp = (
        db.query(Employee)
        .filter(Employee.employee_code == payload.employee_code)
        .first()
    )
    if not emp:
        raise HTTPException(status_code=404, detail="Colaborador não encontrado")

    session = FridgeSession(
        employee_id=emp.id,
        device_id=payload.device_id,
        opened_at=now(),
        status="open",
    )
    db.add(session)
    db.commit()
    db.refresh(session)

    return SessionStartResponse(
        success=True,
        session_id=session.id,
        employee=EmployeeOut.from_orm(emp),
    )


@app.post("/api/session_close")
def session_close(payload: SessionCloseRequest, db: Session = Depends(get_db)):
    session = db.query(FridgeSession).filter(
        FridgeSession.id == payload.session_id
    ).first()
    if not session:
        raise HTTPException(status_code=404, detail="Sessão não encontrada")

    session.closed_at = now()
    session.capture_before_id = payload.capture_before_id
    session.capture_after_id = payload.capture_after_id
    session.status = "closed"
    db.add(session)

    consumed = []
    if payload.capture_before_id and payload.capture_after_id:
        # monta dicionários label -> quantidade
        before_items = {
            i.label: i.quantity
            for i in db.query(VisionItem)
            .filter(VisionItem.capture_id == payload.capture_before_id)
            .all()
        }
        after_items = {
            i.label: i.quantity
            for i in db.query(VisionItem)
            .filter(VisionItem.capture_id == payload.capture_after_id)
            .all()
        }

        for label, qty_before in before_items.items():
            qty_after = after_items.get(label, 0)
            delta = qty_before - qty_after
            if delta > 0:
                consumed.append({"label": label, "quantity": delta})
                ev = ConsumptionEvent(
                    session_id=session.id,
                    employee_id=session.employee_id,
                    product_label=label,
                    quantity=delta,
                    created_at=now(),
                )
                db.add(ev)

    db.commit()

    return {
        "success": True,
        "session_id": session.id,
        "consumed": consumed,
    }


@app.get("/api/sessions_list", response_model=dict)
def sessions_list(employee_code: Optional[str] = None, db: Session = Depends(get_db)):
    q = (
        db.query(
            FridgeSession,
            Employee.employee_code,
            Employee.name.label("employee_name"),
        )
        .join(Employee, FridgeSession.employee_id == Employee.id)
        .order_by(FridgeSession.opened_at.desc())
    )

    if employee_code:
        q = q.filter(Employee.employee_code == employee_code)

    rows = q.all()
    result: List[SessionListItem] = []
    for s, code, emp_name in rows:
        result.append(
            SessionListItem(
                id=s.id,
                employee_id=s.employee_id,
                employee_code=code,
                employee_name=emp_name,
                device_id=s.device_id,
                opened_at=s.opened_at,
                closed_at=s.closed_at,
                capture_before_id=s.capture_before_id,
                capture_after_id=s.capture_after_id,
                status=s.status,
                notes=s.notes,
            )
        )

    return {"success": True, "sessions": result}
