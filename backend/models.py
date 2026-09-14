from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Boolean,
    Numeric,
    DateTime,
    ForeignKey,
    UniqueConstraint,
)

from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from database import Base


class Product(Base):

    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)

    shopify_product_id = Column(
        String(100),
        unique=True,
        nullable=False,
        index=True
    )

    title = Column(String(500), nullable=False)

    handle = Column(String(500), index=True)

    description = Column(Text)

    vendor = Column(String(255))

    product_type = Column(String(255))

    status = Column(String(50))

    category = Column(String(255))

    tags = Column(Text)

    product_url = Column(String(1000))

    image_url = Column(String(2000))

    created_at = Column(DateTime(timezone=True))

    updated_at = Column(DateTime(timezone=True))

    synced_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )

    variants = relationship(
        "ProductVariant",
        back_populates="product",
        cascade="all, delete-orphan"
    )


class ProductVariant(Base):

    __tablename__ = "product_variants"

    id = Column(Integer, primary_key=True, index=True)

    shopify_variant_id = Column(
        String(100),
        unique=True,
        nullable=False,
        index=True
    )

    product_id = Column(
        Integer,
        ForeignKey("products.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    title = Column(String(500))

    sku = Column(String(255), index=True)

    price = Column(Numeric(12, 2))

    compare_at_price = Column(Numeric(12, 2))

    size = Column(String(100))

    color = Column(String(100))

    inventory_quantity = Column(Integer)

    available = Column(Boolean, default=False)

    image_url = Column(String(2000))

    created_at = Column(DateTime(timezone=True))

    updated_at = Column(DateTime(timezone=True))

    product = relationship(
        "Product",
        back_populates="variants"
    )

    __table_args__ = (
        UniqueConstraint(
            "shopify_variant_id",
            name="uq_shopify_variant_id"
        ),
    )