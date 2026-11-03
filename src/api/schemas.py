"""Pydantic-схемы запросов и ответов API.

Ограничения полей повторяют правила Great Expectations: некорректный запрос
(например, возраст 150 или неизвестный код образования) отклоняется с
кодом 422 ещё до обращения к модели.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

PAY_STATUS_DESCRIPTION = (
    "Repayment status: -2 no consumption, -1 paid in full, 0 revolving credit, "
    "1..9 payment delay in months"
)

EXAMPLE_CLIENT = {
    "LIMIT_BAL": 50000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 35,
    "PAY_0": 2,
    "PAY_2": 2,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT1": 46990,
    "BILL_AMT2": 48233,
    "BILL_AMT3": 49291,
    "BILL_AMT4": 28314,
    "BILL_AMT5": 28959,
    "BILL_AMT6": 29547,
    "PAY_AMT1": 2000,
    "PAY_AMT2": 2019,
    "PAY_AMT3": 1200,
    "PAY_AMT4": 1100,
    "PAY_AMT5": 1069,
    "PAY_AMT6": 1000,
}

_PAY = {"ge": -2, "le": 9, "description": PAY_STATUS_DESCRIPTION}
_BILL = {"ge": -1_000_000, "le": 10_000_000}
_PAYMENT = {"ge": 0, "le": 10_000_000}


class ClientData(BaseModel):
    """Анкета и платёжная история клиента за 6 месяцев (апрель–сентябрь 2005)."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_CLIENT]})

    LIMIT_BAL: float = Field(ge=1_000, le=5_000_000, description="Credit limit, NT dollars")
    SEX: int = Field(ge=1, le=2, description="1 = male, 2 = female")
    EDUCATION: int = Field(
        ge=0, le=6, description="1 graduate school, 2 university, 3 high school, 4/0/5/6 other"
    )
    MARRIAGE: int = Field(ge=0, le=3, description="1 married, 2 single, 3/0 other")
    AGE: int = Field(ge=18, le=100, description="Age in years")
    PAY_0: int = Field(**_PAY)
    PAY_2: int = Field(**_PAY)
    PAY_3: int = Field(**_PAY)
    PAY_4: int = Field(**_PAY)
    PAY_5: int = Field(**_PAY)
    PAY_6: int = Field(**_PAY)
    BILL_AMT1: float = Field(**_BILL, description="Bill statement, September")
    BILL_AMT2: float = Field(**_BILL, description="Bill statement, August")
    BILL_AMT3: float = Field(**_BILL, description="Bill statement, July")
    BILL_AMT4: float = Field(**_BILL, description="Bill statement, June")
    BILL_AMT5: float = Field(**_BILL, description="Bill statement, May")
    BILL_AMT6: float = Field(**_BILL, description="Bill statement, April")
    PAY_AMT1: float = Field(**_PAYMENT, description="Amount paid in September")
    PAY_AMT2: float = Field(**_PAYMENT, description="Amount paid in August")
    PAY_AMT3: float = Field(**_PAYMENT, description="Amount paid in July")
    PAY_AMT4: float = Field(**_PAYMENT, description="Amount paid in June")
    PAY_AMT5: float = Field(**_PAYMENT, description="Amount paid in May")
    PAY_AMT6: float = Field(**_PAYMENT, description="Amount paid in April")


class PredictionResponse(BaseModel):
    default_prediction: int = Field(description="1 = default expected next month, 0 = no default")
    default_probability: float = Field(ge=0, le=1, description="Probability of default (PD)")
    threshold: float = Field(description="Decision threshold applied to the probability")
    model_name: str


class BatchRequest(BaseModel):
    clients: list[ClientData] = Field(min_length=1, max_length=10_000)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
