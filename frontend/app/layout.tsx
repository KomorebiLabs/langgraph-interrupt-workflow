/**
 * 根布局：主页与 /approval 共享 globals.css、Tailwind 基础样式和 antialiased body。
 * metadata 是 Next.js 文档契约；children 由 App Router 注入，布局不参与业务状态。
 */
import "./globals.css";

// 浏览器标题与描述属于应用壳层，不携带 workflow/approval 数据。
export const metadata = {
  title: "AI Research Assistant",
  description: "Intelligent research with interactive guidance using LangGraph",
};

export default function RootLayout({
  // html lang 当前为英文，因为界面文案和协议提示均以英文呈现。
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
