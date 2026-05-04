import { Navigate, Route, Routes } from "react-router-dom";
import { ForecastPage } from "../pages/ForecastPage";
import { SessionsPage } from "../pages/SessionsPage";
import { DecisionSupportPage } from "../pages/DecisionSupportPage";
import { OpsPage } from "../pages/OpsPage";
import { NotFoundPage } from "../pages/NotFoundPage";

export function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/forecast" replace />} />
      <Route path="/forecast" element={<ForecastPage />} />
      <Route path="/decision-support" element={<DecisionSupportPage />} />
      <Route path="/sessions" element={<SessionsPage />} />
      <Route path="/ops" element={<OpsPage />} />
      <Route path="*" element={<NotFoundPage />} />
    </Routes>
  );
}
