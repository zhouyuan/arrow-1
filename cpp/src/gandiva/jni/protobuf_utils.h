#include <google/protobuf/io/coded_stream.h>

#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <arrow/builder.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>

#include <gandiva/arrow.h>
#include <gandiva/gandiva_aliases.h>
#include <gandiva/tree_expr_builder.h>

#include "Types.pb.h"

using gandiva::ConditionPtr;
using gandiva::DataTypePtr;
using gandiva::ExpressionPtr;
using gandiva::ExpressionVector;
using gandiva::FieldPtr;
using gandiva::FieldVector;
using gandiva::NodePtr;
using gandiva::NodeVector;
using gandiva::SchemaPtr;
using gandiva::Status;
using gandiva::TreeExprBuilder;

using gandiva::ArrayDataVector;

// forward declarations
NodePtr ProtoTypeToNode(const types::TreeNode& node);

DataTypePtr ProtoTypeToTime32(const types::ExtGandivaType& ext_type);
DataTypePtr ProtoTypeToTime64(const types::ExtGandivaType& ext_type);
DataTypePtr ProtoTypeToTimestamp(const types::ExtGandivaType& ext_type);
DataTypePtr ProtoTypeToDataType(const types::ExtGandivaType& ext_type);
FieldPtr ProtoTypeToField(const types::Field& f);
NodePtr ProtoTypeToFieldNode(const types::FieldNode& node);
NodePtr ProtoTypeToFnNode(const types::FunctionNode& node);
NodePtr ProtoTypeToIfNode(const types::IfNode& node);
NodePtr ProtoTypeToAndNode(const types::AndNode& node);
NodePtr ProtoTypeToOrNode(const types::OrNode& node);
NodePtr ProtoTypeToInNode(const types::InNode& node);
NodePtr ProtoTypeToNullNode(const types::NullNode& node);
NodePtr ProtoTypeToNode(const types::TreeNode& node);
ExpressionPtr ProtoTypeToExpression(const types::ExpressionRoot& root);
ConditionPtr ProtoTypeToCondition(const types::Condition& condition);
SchemaPtr ProtoTypeToSchema(const types::Schema& schema);
// Common for both projector and filters.
bool ParseProtobuf(uint8_t* buf, int bufLen, google::protobuf::Message* msg);
